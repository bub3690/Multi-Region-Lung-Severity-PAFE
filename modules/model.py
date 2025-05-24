import torch
import torch.nn as nn
import torchvision

# 현재 폴더 .을 path로 추가

from .vit import PatchEmbedding, TransformerEncoderBlock
from .backbones import model_dict
import torch.nn.functional as F
from .stn import SpatialTransformer
from segementation.HybridGNet2IGSC import HybridGNet
from seg_utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart
import scipy.sparse as sp



class CNN(nn.Module):
    def __init__(self,backbone,num_classes=4,num_regions=6):
        super(CNN, self).__init__()
        self.backbone, self.num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
    
        # remove GAP, FC layer
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16
        
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.num_classes)
        
        
    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]

        if self.num_regions == 6:
            boxes = [torch.tensor([0, 0, 0.4, 0.5]),  # A (1)
                    torch.tensor([0.3, 0, 0.7, 0.5]),  # B (2)
                    torch.tensor([0.6, 0, 1, 0.5]),  # C (3)
                    torch.tensor([0, 0.5, 0.4, 1]),  # D (4)
                    torch.tensor([0.3, 0.5, 0.7, 1]),  # E (5)
                    torch.tensor([0.6, 0.5, 1, 1])  # F (6)
                    ]
        elif self.num_regions == 4:
            boxes = [torch.tensor([0, 0, 0.5, 0.5]),  # A (1)
                    torch.tensor([0.5, 0, 1, 0.5]),  # B (2)
                    torch.tensor([0, 0.5, 0.5, 1]),  # C (3)
                    torch.tensor([0.5, 0.5, 1, 1])  # D (4)
                    ]

        out = []
        for b in boxes:
            car = F.interpolate(
                x[:, :,
                  int(b[0]*x.shape[2]):int(b[2]*x.shape[2]),
                  int(b[1]*x.shape[3]):int(b[3]*x.shape[3])
                  ],
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            out.append(car)
        return torch.stack(out, dim=1)
    
    def forward(self, x, embedding=False):        
        layer5_out = self.backbone(x)
        layer5_out = self.pool_rois(layer5_out) # torch.Size([1, 6, 512, 16, 16])
        
        
        b = layer5_out
        #print(b.shape)
        retina_net_class = []
        for i in range(self.num_regions):
            pred_i = b[:,i,:,:,:]
            #print('selection : ',pred_i.shape)
            pred_i = self.score_GAP(pred_i)
            pred_i = pred_i.view(-1,self.num_features)
            #print("gap : ",pred_i.shape)
            
            if embedding == False:
                pred_i = self.fc(pred_i)
                #print(pred_i.shape)
                pred_i = F.softmax(pred_i, dim=1)
            retina_net_class.append(pred_i)
        retina_net_class = torch.stack(retina_net_class,dim=1)
        return retina_net_class




class Hybrid(nn.Module):
    def __init__(self,backbone,num_classes=4,num_regions=6):
        super(Hybrid, self).__init__()
        self.backbone,num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        
        # backbone에서 GAP제거
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16
        
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        
        ###
        # https://github.com/FrancescoSaverioZuppichini/ViT
        # Transformer hybrid network
        
        self.patch_emb = PatchEmbedding(in_channels=num_features, patch_size=1, emb_size=768, img_size=self.img_size)
        
        # 2개의 transformer block
        self.transformer  = TransformerEncoderBlock()     
        self.fc = nn.Linear(768, self.num_classes)
    
        
    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]

        if self.num_regions == 6:
            boxes = [torch.tensor([0, 0, 0.4, 0.5]),  # A (1)
                    torch.tensor([0.3, 0, 0.7, 0.5]),  # B (2)
                    torch.tensor([0.6, 0, 1, 0.5]),  # C (3)
                    torch.tensor([0, 0.5, 0.4, 1]),  # D (4)
                    torch.tensor([0.3, 0.5, 0.7, 1]),  # E (5)
                    torch.tensor([0.6, 0.5, 1, 1])  # F (6)
                    ]
        elif self.num_regions == 4:
            boxes = [torch.tensor([0, 0, 0.5, 0.5]),  # A (1)
                    torch.tensor([0.5, 0, 1, 0.5]),  # B (2)
                    torch.tensor([0, 0.5, 0.5, 1]),  # C (3)
                    torch.tensor([0.5, 0.5, 1, 1])  # D (4)
                    ]
        
        out = []
        for b in boxes:
            car = F.interpolate(
                x[:, :,
                  int(b[0]*x.shape[2]):int(b[2]*x.shape[2]),
                  int(b[1]*x.shape[3]):int(b[3]*x.shape[3])
                  ],
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            out.append(car)
        return torch.stack(out, dim=1)

    
    def forward(self, x, embedding=False):
        layer5_out = self.backbone(x)
        
        
        #print(b.shape)
        
        ###
        #여기서 vit 작업.
        # print('before patch : ',layer5_out.shape)
        layer5_out = self.patch_emb(layer5_out)
        
        ###
        layer5_out = self.transformer(layer5_out)
        layer5_out = layer5_out.reshape(-1,16,16,768)
        layer5_out = layer5_out.permute(0,3,1,2)
        # print('after transformer : ',layer5_out.shape)
        
        
        b = self.pool_rois(layer5_out) # torch.Size([1, 6, 512, 16, 16])
        
        retina_net_class = []
        for i in range(self.num_regions):
            pred_i = b[:,i,:,:,:]
            #print('selection : ',pred_i.shape)
            pred_i = self.score_GAP(pred_i)
            pred_i = pred_i.view(-1,768)
            #print("gap : ",pred_i.shape)
            
            if embedding == False:                
                #print('transformer : ',pred_i.shape)
                pred_i = self.fc(pred_i)
                #print(pred_i.shape)
                pred_i = F.softmax(pred_i, dim=1)
            retina_net_class.append(pred_i)
        retina_net_class = torch.stack(retina_net_class,dim=1)
        return retina_net_class

############
# Updated 2025.02.19
# Hybrid end-to-end model (using pretrained Spatial Normalization network)
############








class Hybrid_e2e(nn.Module):
    def __init__(self,backbone,num_classes=4,num_regions=6):
        super(Hybrid, self).__init__()
        
        ##
        # Spatial Normalzation network
        
        
        
        
        ##
        
        
        self.backbone,num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        
        # Remove GAP, FC layer
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16
        
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        
        ###
        # https://github.com/FrancescoSaverioZuppichini/ViT
        # Transformer hybrid network
        
        self.patch_emb = PatchEmbedding(in_channels=num_features, patch_size=1, emb_size=768, img_size=self.img_size)
        
        # 2개의 transformer block
        self.transformer  = TransformerEncoderBlock()     
        self.fc = nn.Linear(768, self.num_classes)
    
        
    def pool_rois(self, x, crop_size=None):
        if crop_size is None:
            crop_size = x.shape[2:4]

        if self.num_regions == 6:
            boxes = [torch.tensor([0, 0, 0.4, 0.5]),  # A (1)
                    torch.tensor([0.3, 0, 0.7, 0.5]),  # B (2)
                    torch.tensor([0.6, 0, 1, 0.5]),  # C (3)
                    torch.tensor([0, 0.5, 0.4, 1]),  # D (4)
                    torch.tensor([0.3, 0.5, 0.7, 1]),  # E (5)
                    torch.tensor([0.6, 0.5, 1, 1])  # F (6)
                    ]
        elif self.num_regions == 4:
            boxes = [torch.tensor([0, 0, 0.5, 0.5]),  # A (1)
                    torch.tensor([0.5, 0, 1, 0.5]),  # B (2)
                    torch.tensor([0, 0.5, 0.5, 1]),  # C (3)
                    torch.tensor([0.5, 0.5, 1, 1])  # D (4)
                    ]
        
        out = []
        for b in boxes:
            car = F.interpolate(
                x[:, :,
                  int(b[0]*x.shape[2]):int(b[2]*x.shape[2]),
                  int(b[1]*x.shape[3]):int(b[3]*x.shape[3])
                  ],
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            out.append(car)
        return torch.stack(out, dim=1)

    
    def forward(self, x, embedding=False):
        layer5_out = self.backbone(x)
        
        
        #print(b.shape)
        
        ###
        #여기서 vit 작업.
        # print('before patch : ',layer5_out.shape)
        layer5_out = self.patch_emb(layer5_out)
        
        ###
        layer5_out = self.transformer(layer5_out)
        layer5_out = layer5_out.reshape(-1,16,16,768)
        layer5_out = layer5_out.permute(0,3,1,2)
        # print('after transformer : ',layer5_out.shape)
        
        
        b = self.pool_rois(layer5_out) # torch.Size([1, 6, 512, 16, 16])
        
        retina_net_class = []
        for i in range(self.num_regions):
            pred_i = b[:,i,:,:,:]
            #print('selection : ',pred_i.shape)
            pred_i = self.score_GAP(pred_i)
            pred_i = pred_i.view(-1,768)
            #print("gap : ",pred_i.shape)
            
            if embedding == False:                
                #print('transformer : ',pred_i.shape)
                pred_i = self.fc(pred_i)
                #print(pred_i.shape)
                pred_i = F.softmax(pred_i, dim=1)
            retina_net_class.append(pred_i)
        retina_net_class = torch.stack(retina_net_class,dim=1)
        return retina_net_class




class Hybrid_e2e_stn(nn.Module):
    def __init__(self, backbone, num_classes=4, num_regions=6, seg_checkpoint=None, stn_checkpoint=None):
        super(Hybrid_e2e_stn, self).__init__()
        self.backbone, num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
        
        # Segmentation model setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        A, AD, D, U = genMatrixesLungsHeart()
        N1 = A.shape[0]
        N2 = AD.shape[0]

        A = sp.csc_matrix(A).tocoo()
        AD = sp.csc_matrix(AD).tocoo()
        D = sp.csc_matrix(D).tocoo()
        U = sp.csc_matrix(U).tocoo()

        D_ = [D.copy()]
        U_ = [U.copy()]

        config = {}
        config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
        A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
        
        A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(self.device) for x in X] for X in (A_, D_, U_))

        config['latents'] = 64
        config['inputsize'] = 1024

        f = 32
        config['filters'] = [2, f, f, f, f//2, f//2, f//2]
        config['skip_features'] = f

        self.segmentation = HybridGNet(config.copy(), D_t, U_t, A_t).to(self.device)
        # Load pre-trained weights
        if seg_checkpoint is None:
            seg_checkpoint = "checkpoint/seg/seg_weights.pt"
        seg_weights = torch.load(seg_checkpoint, map_location=self.device)
        self.segmentation.load_state_dict(seg_weights)
        # Freeze segmentation model
        for param in self.segmentation.parameters():
            param.requires_grad = False
        
        # STN module with configuration
        self.stn = SpatialTransformer(
            in_shape=(1, 512, 512),  # Input shape (segmentation mask)
            mask_resize=512,         # Output size
            dense_neurons=50,        # Number of neurons in dense layer
            freeze_align_model=False # Allow training of alignment
        )
        
        # Load STN weights if provided
        if stn_checkpoint is not None:
            stn_weights = torch.load(stn_checkpoint, map_location=self.device)
            self.stn.load_state_dict(stn_weights)
        
        # backbone에서 GAP제거
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16
        
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        # Transformer hybrid network
        self.patch_emb = PatchEmbedding(in_channels=num_features, patch_size=1, emb_size=768, img_size=self.img_size)
        self.transformer = TransformerEncoderBlock()
        self.fc = nn.Linear(768, self.num_classes)

    def pool_rois(self, x, crop_size=None):
        """ROI 영역을 추출하고 풀링합니다."""
        if crop_size is None:
            crop_size = x.shape[2:4]

        if self.num_regions == 6:
            boxes = [torch.tensor([0, 0, 0.4, 0.5]),  # A (1)
                    torch.tensor([0.3, 0, 0.7, 0.5]),  # B (2)
                    torch.tensor([0.6, 0, 1, 0.5]),  # C (3)
                    torch.tensor([0, 0.5, 0.4, 1]),  # D (4)
                    torch.tensor([0.3, 0.5, 0.7, 1]),  # E (5)
                    torch.tensor([0.6, 0.5, 1, 1])  # F (6)
                    ]
        elif self.num_regions == 4:
            boxes = [torch.tensor([0, 0, 0.5, 0.5]),  # A (1)
                    torch.tensor([0.5, 0, 1, 0.5]),  # B (2)
                    torch.tensor([0, 0.5, 0.5, 1]),  # C (3)
                    torch.tensor([0.5, 0.5, 1, 1])  # D (4)
                    ]

        out = []
        for b in boxes:
            car = F.interpolate(
                x[:, :,
                  int(b[0]*x.shape[2]):int(b[2]*x.shape[2]),
                  int(b[1]*x.shape[3]):int(b[3]*x.shape[3])
                  ],
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            out.append(car)
        return torch.stack(out, dim=1)

    def create_mask_from_landmarks(self, landmarks, pos1, pos2, size):
        """랜드마크 포인트들로부터 세그멘테이션 마스크를 생성합니다."""
        batch_size = landmarks.shape[0]
        masks = []
        
        for i in range(batch_size):
            mask = torch.zeros((1, size[0], size[1]), device=self.device)
            
            # 오른쪽 폐 (0-43), 왼쪽 폐 (44-93), 심장 (94-끝)
            rl_points = landmarks[i, 0:44].long()
            ll_points = landmarks[i, 44:94].long()
            heart_points = landmarks[i, 94:].long()
            
            # 각 영역을 폴리곤으로 채우기
            for points in [rl_points, ll_points, heart_points]:
                # 포인트들을 이미지 크기에 맞게 스케일링
                points = (points * size[0]).long()
                points = points.clamp(0, size[0]-1)
                
                # 폴리곤 생성
                for j in range(points.shape[0]-1):
                    p1 = points[j]
                    p2 = points[j+1]
                    # 두 점 사이에 선 그리기
                    mask[0] = mask[0].index_put_((torch.arange(p1[1], p2[1]).long(),
                                                torch.arange(p1[0], p2[0]).long()),
                                               torch.ones_like(torch.arange(p1[0], p2[0]), device=self.device))
            
            masks.append(mask)
        
        return torch.stack(masks)

    def forward(self, x):
        # Get segmentation mask using HybridGNet2IGSC
        with torch.no_grad():
            # Ensure input is properly sized for segmentation model
            if x.shape[-1] != 1024 or x.shape[-2] != 1024:
                x_seg = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
            else:
                x_seg = x
            
            # Convert to single channel if input is RGB
            if x_seg.shape[1] == 3:
                x_seg = 0.299 * x_seg[:, 0:1] + 0.587 * x_seg[:, 1:2] + 0.114 * x_seg[:, 2:3]
            
            # Get segmentation output (landmarks)
            landmarks, pos1, pos2 = self.segmentation(x_seg)
            
            # Convert landmarks to dense mask
            seg_mask = self.create_mask_from_landmarks(landmarks, pos1, pos2, 
                                                     size=(x.shape[2], x.shape[3]))
        
        # Apply STN on segmentation mask
        transformed_mask = self.stn(seg_mask)
        
        # Apply same transformation to original image
        grid = self.stn.get_last_grid()
        transformed_img = F.grid_sample(x, grid, align_corners=False)
        
        # Extract features using backbone
        layer5_out = self.backbone(transformed_img)
        
        # Apply transformer
        layer5_out = self.patch_emb(layer5_out)
        layer5_out = self.transformer(layer5_out)
        layer5_out = layer5_out.reshape(-1, 16, 16, 768)
        layer5_out = layer5_out.permute(0, 3, 1, 2)
        
        # Pool ROIs and classify
        layer5_out = self.pool_rois(layer5_out)
        
        retina_net_class = []
        for i in range(self.num_regions):
            pred_i = layer5_out[:, i, :, :, :]
            pred_i = self.score_GAP(pred_i)
            pred_i = pred_i.view(-1, 768)
            
            pred_i = self.fc(pred_i)
            pred_i = F.softmax(pred_i, dim=1)
            retina_net_class.append(pred_i)
        
        retina_net_class = torch.stack(retina_net_class, dim=1)
        return retina_net_class, transformed_img, transformed_mask




if __name__ == "__main__":
    #model = CNN('densenet121')
    

    model = CNN('densenet121',num_classes=5,num_regions=4)
    
    sample = torch.randn(8, 3, 512, 512)
    print(model(sample).shape)
    
    #print(model)
