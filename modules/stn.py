import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    def __init__(self, in_shape=(1, 512, 512), mask_resize=512, dense_neurons=50, freeze_align_model=False):
        super(SpatialTransformer, self).__init__()
        
        assert not in_shape[1] % mask_resize, "The STN size must be a multiple of mask size"
        trainable = not freeze_align_model
        
        self._last_grid = None

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

    def get_last_grid(self):
        """Return the last computed transformation grid"""
        if self._last_grid is None:
            raise RuntimeError("No grid has been computed yet. Run forward pass first.")
        return self._last_grid

    def forward(self, x):
        # Handle channel dimension mismatch
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)  # Convert RGB to grayscale if needed
            
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
        self._last_grid = grid  # Store the grid
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x 