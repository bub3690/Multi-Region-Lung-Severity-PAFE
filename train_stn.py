import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import cv2


from glob import glob


# Augmentation with Albumentations
train_transform = A.Compose([
    A.Rotate(limit=25, p=0.8),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.8),
    A.ElasticTransform(alpha=60, sigma=12, p=0.2),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.2),
    A.Resize(512, 512),  # Rescale all images to a fixed size
    ToTensorV2()
])

class LungSegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pre_mask = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=pre_mask)
            pre_mask = augmented['image']
        
        # 0~1로 정규화
        pre_mask = pre_mask / 255.0
        mask = mask / 255.0
        
        
        return pre_mask, mask



pre_mask = glob('./data/brixia/processed_masks/*.png')
post_mask = glob("./data/brixia/registration/mask/*.png")

sorted(pre_mask), sorted(post_mask)
# 두 데이터들이 같은지 확인. 파일이름이같은지 100개만 확인.
for i in range(100):
    if pre_mask[i].split('/')[-1][:-4] != post_mask[i].split('/')[-1][:-4]:
        print("Different file names")
        break


# Define dataset and dataloader
train_dataset = LungSegmentationDataset(pre_mask,post_mask, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.contiguous()
        targets = targets.contiguous()
        intersection = (inputs * targets)
        intersection = intersection.sum(dim=2).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=2).sum(dim=1) + targets.sum(dim=2).sum(dim=1) + self.smooth)
        return 1 - dice.mean()

# Spatial Transformer Network (STN)
class STN(nn.Module):
    def __init__(self, in_shape=(1,512,512), mask_resize: int = 512,
                 dense_neurons=50, freeze_align_model=False):
        super(STN, self).__init__()
        
        assert not in_shape[1] % mask_resize, "The STN size must be a multiple of mask size"
        trainable = not freeze_align_model

        # MaxPooling for input adaptation
        self.pool1 = nn.MaxPool2d(kernel_size=(in_shape[1] // mask_resize, in_shape[2] // mask_resize))

        # Convolutional and pooling layers
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=in_shape[0], out_channels=20, kernel_size=5, stride=1)
        if not trainable:
            self.conv1.requires_grad_(False)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1)
        if not trainable:
            self.conv2.requires_grad_(False)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=self.calculate_flatten_size(in_shape), out_features=dense_neurons)
        if not trainable:
            self.fc1.requires_grad_(False)
        
        self.relu = nn.ReLU()

        # Final alignment layer (6 affine parameters)
        self.fc2 = nn.Linear(in_features=dense_neurons, out_features=6)
        if not trainable:
            self.fc2.requires_grad_(False)
        
        # Initialize the alignment layer to the identity transformation
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def calculate_flatten_size(self, in_shape):
        # Dummy input to calculate the size after convolution and pooling
        dummy_input = torch.randn(1, in_shape[0], in_shape[1], in_shape[2])
        x = self.pool1(dummy_input)
        x = self.pool2(self.conv1(x))
        x = self.pool3(self.conv2(x))
        return x.numel()

    def forward(self, x):
        # Max Pooling and Convolutional layers
        xs = self.pool1(x)
        xs = self.pool2(self.conv1(xs))
        xs = self.pool3(self.conv2(xs))
        
        # Flatten the feature maps
        xs = self.flatten(xs)

        # Fully connected layers and ReLU
        xs = self.fc1(xs)
        xs = self.relu(xs)
        
        # Output the 6 affine parameters
        theta = self.fc2(xs)
        theta = theta.view(-1, 2, 3)  # Reshape to 2x3 affine matrix
        # Generate affine grid and perform sampling
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x

# Training loop setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stn_model = STN().to(device)

# Loss function and optimizer
criterion = F.mse_loss  # For simplicity, we assume reconstruction loss for alignment
optimizer = torch.optim.Adam(stn_model.parameters(), lr=0.0001)

from matplotlib import pyplot as plt
# dataset하나 출력하고 이미지 저장.
for images, masks in train_loader:
    print(images.shape, masks.shape)
    print(images.max(), masks.max())
    print(images.min(), masks.min())
    # save
    plt.imshow(images[0].squeeze(0).cpu().numpy(), cmap='gray')
    plt.savefig('original.png')
    plt.imshow(masks[0].squeeze(0).cpu().numpy(), cmap='gray')
    plt.savefig('mask.png')
    break


# Training loop
for epoch in range(10):  # Number of epochs
    stn_model.train()
    
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        optimizer.zero_grad()

        # Forward pass through STN
        aligned_images = stn_model(images)
        
        aligned_images = aligned_images.squeeze(1)  # Remove channel dimension
        

        # Loss: compare aligned image with original segmentation (for alignment)
        loss = criterion(aligned_images, masks)  # Assuming the target is to align with the original input
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], Loss: {running_loss / len(train_loader)}")

#Save checkpoint
# /hdd/project/cxr_haziness/Uncertainty/checkpoint/stn/checkpoint.pth
torch.save(stn_model.state_dict(), './checkpoint/stn/checkpoint.pth')


