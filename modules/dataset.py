from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from skimage import exposure
from PIL import Image
import cv2
import torch
import numpy as np
import pandas as pd

class BRIXIA(Dataset):
    def __init__(self, filename_list, label_list, label_df, prefix, sum_lung=True,transform=None,augmentation=None,train=True):
        self.filename_list = filename_list
        self.label = label_list
        self.label_df = label_df # pandas dataframe
        self.to_tensor = transforms.ToTensor()
        self.prefix = prefix
        self.train = train
        
        # self.prefix 파일 경로에서 마지막 4번째,3번째 폴더를 제거
        #self.prefix_mask = "/hdd/project/cxr_haziness/data/0407_0608_mask/processed_images/"
        
        self.transform = transform # 이미지 사이즈 변환, 정규화 등 기본 변환
        self.augmentation = augmentation # 온라인 데이터 증강 방법들
        
        self.strong_transform = transforms.Compose(
                [
                # Chexpert에서 사용한 augmentation
                
                #transforms.ToPILImage(),
                #transforms.RandomHorizontalFlip(), # 일단 빼고, 나중에 레이블 반영해서 추가하기.
                # distortion
                # 회전
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.8),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)], p=0.8),
                # random sharpness
                transforms.RandomAdjustSharpness(sharpness_factor=0.0,p=0.8),
                
                # random affine
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05,0.05))], p=0.8),
                
                
                ]
            )
        
    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        
        full_filename = self.label_df[self.label_df["Filename"]==self.filename_list[idx]]["Filename"].values[0]
        # full_filename to png
        full_filename = full_filename[:-4]+".png"
        
        file_path = self.prefix + full_filename

        image = Image.open( file_path )
        ##
        
        now_label = torch.tensor(self.label[idx],dtype=torch.int64)
        now_label_float  = torch.tensor(self.label[idx],dtype=torch.float32)
        #augmentation. Albumentation의 데이터 증강은 numpy에만 적용됨.
        
        if self.augmentation:
            image=self.augmentation(image=image)
            image=image['image']
        
        
        # 전처리
        # image = np.array(image)
        # image = exposure.equalize_adapthist(image/255.0)
        ##         
        
        image= self.to_tensor(image)
        
        if self.transform: 
            image = self.transform(image).type(torch.float32)
            if self.train:
                image = self.strong_transform(image)            
        
        # 3채널
        if image.shape[0] != 3:
            image = image.expand(3,-1,-1)
        
        return image, now_label,now_label_float, self.filename_list[idx]
    
    
class InhaPrivate(Dataset):
    def __init__(self, filename_list, label_list, label_df, prefix, sum_lung=False,transform=None,augmentation=None,train=True):
        # #### case 3 ####
        self.filename_list = filename_list
        self.label = label_list
        self.label_df = label_df # pandas dataframe
        self.to_tensor = transforms.ToTensor()
        self.prefix = prefix
        self.sum_lung = sum_lung # 폐(상하)를 합쳐서 레이블링 할지.
        self.train = train
        
        self.transform = transform # 이미지 사이즈 변환, 정규화 등 기본 변환
        self.augmentation = augmentation # 온라인 데이터 증강 방법들
        
        self.strong_transform = transforms.Compose(
                [
                # Chexpert에서 사용한 augmentation
                
                #transforms.ToPILImage(),
                #transforms.RandomHorizontalFlip(), # 일단 빼고, 나중에 레이블 반영해서 추가하기.
                # distortion
                # 회전
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.8),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)], p=0.8),
                # random sharpness
                transforms.RandomAdjustSharpness(sharpness_factor=0.0,p=0.8),
                
                # random affine
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05,0.05))], p=0.8),
                
                
                ]
            )

    def __len__(self):
        return len(self.filename_list)
        

    def __getitem__(self, idx):
        
        #full_filename to png
        full_filename = str(self.filename_list[idx])+".png"

        file_path = self.prefix + full_filename

        image = Image.open( file_path )

        now_label_list = self.label[idx]
        
        # now_label 배열의 값들을 1,2,3,4,5 에서 0,1,2,3,4로 바꿔준다.
        now_label_list = [x - 1 for x in now_label_list]
        
        # RT,RB,LT,LB가 되게 해야함.
        # 순서 변경. RT, LT, RB, LB -> R(0,2), L(1,3)
        
        
        if self.sum_lung:
            # RT, LT, RB, LB -> R(0,2), L(1,3)
            now_label = torch.tensor([
                                    torch.max( torch.tensor( [int(now_label_list[0]),  int(now_label_list[2]) ] )),
                                    torch.max( torch.tensor( [int(now_label_list[1]),  int(now_label_list[3]) ] ))
                                    ],dtype=torch.long)
        else:
            now_label = torch.tensor([int(now_label_list[0]),  
                                      int(now_label_list[2]),
                                      int(now_label_list[1]),
                                      int(now_label_list[3])],dtype=torch.long)
            now_label_float = torch.tensor([int(now_label_list[0]),  
                                      int(now_label_list[2]),
                                      int(now_label_list[1]),
                                      int(now_label_list[3])],dtype=torch.float32)            
        
        #print(now_label)
        #augmentation. Albumentation의 데이터 증강은 numpy에만 적용됨.
        if self.augmentation:
            image=self.augmentation(image = image)
            image=image['image']
        
        image= self.to_tensor(image)
        
        if self.transform: 
            image = self.transform(image).type(torch.float32)
            if self.train:
                image = self.strong_transform(image)    
        
        if image.shape[0] != 3:
            image = image.expand(3,-1,-1)        
        
        
        return image, now_label, now_label_float, str(self.filename_list[idx]) 
    
    
if __name__ == '__main__':
    label_df = pd.read_csv("/hdd/project/cxr_haziness/data/CXR_23_1113/df_preprocess_file_remove_ver2.csv")
    data_set=InhaPrivate(["1"],[[1,2,3,4]],None,"/hdd/project/cxr_haziness/data/CXR_23_1113/registration/images/",sum_lung=False,transform=None,augmentation=None,train=True)
    
    print(data_set.__getitem__(0))
    
    
    