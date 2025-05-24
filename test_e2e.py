import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
import time

from modules.model import Hybrid_e2e_stn
from modules.dataset import BRIXIA

def evaluate(model, test_dataloader, classify_criterion, DEVICE, save_stn_results=False, save_dir=None):
    model.eval()
    test_correct = [0,0,0,0,0,0]
    total_test_loss = 0
    
    all_preds = [[] for _ in range(6)]
    all_labels = [[] for _ in range(6)]
    
    with torch.no_grad():
        for idx, (img, label, label_float, filename) in enumerate(test_dataloader):
            img, label, label_float = img.to(DEVICE), label.to(DEVICE), label_float.to(DEVICE)
            
            # 모델 예측 (이제 변환된 이미지와 마스크도 반환)
            pred, transformed_img, transformed_mask = model(img)
            
            # STN 결과 저장 (처음 5개 이미지에 대해서만)
            if save_stn_results and idx < 5:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # 원본 이미지
                axes[0].imshow(img[0, 0].cpu().numpy(), cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                # 세그멘테이션 마스크
                axes[1].imshow(transformed_mask[0, 0].cpu().numpy(), cmap='gray')
                axes[1].set_title('Segmentation Mask')
                axes[1].axis('off')
                
                # 변환된 마스크
                axes[2].imshow(transformed_mask[0, 0].cpu().numpy(), cmap='gray')
                axes[2].set_title('Transformed Mask')
                axes[2].axis('off')
                
                # 변환된 이미지
                axes[3].imshow(transformed_img[0, 0].cpu().numpy(), cmap='gray')
                axes[3].set_title('Transformed Image')
                axes[3].axis('off')
                
                plt.savefig(os.path.join(save_dir, f'stn_result_{idx}.png'))
                plt.close()
            
            # Loss 및 정확도 계산
            loss_classification = 0
            for i in range(6):
                pos_probs = pred[:,i,:]
                loss_classification += classify_criterion(pos_probs, label[:,i])
                
                prediction = pos_probs.max(1, keepdim=True)[1]
                test_correct[i] += prediction.eq(label[:,i].view_as(prediction)).sum().item()
                
                # 예측값과 라벨 저장
                all_preds[i].extend(prediction.cpu().numpy().flatten())
                all_labels[i].extend(label[:,i].cpu().numpy())
            
            loss_classification = loss_classification / 6
            total_test_loss += loss_classification.item()
            
        total_test_loss = total_test_loss / len(test_dataloader)
        
        # 각 위치별 정확도 계산 및 출력
        test_acc_list = []
        for i in range(6):
            acc = 100. * test_correct[i] / len(test_dataloader.dataset)
            test_acc_list.append(acc)
            print(f"Position {i} Test ACC : {acc:.2f}")
            
            # 각 위치별 혼동 행렬 생성
            if save_dir:
                cm = confusion_matrix(all_labels[i], all_preds[i])
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - Position {i}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(os.path.join(save_dir, f'confusion_matrix_pos_{i}.png'))
                plt.close()
        
        print(f"Mean Test ACC : {np.mean(test_acc_list):.2f}")
        print(f"Test Loss : {total_test_loss:.4f}")
        
        return total_test_loss, test_acc_list, all_preds, all_labels

def main(args):
    # Device 설정
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('Using PyTorch version:', torch.__version__, 'Device:', DEVICE)
    
    # 경로 설정
    base_path = '/multi-region-severity-hybridNet/'
    data_path = './data/'
    
    # 모델 이름 설정
    model_name = f'brixia_{args.arch}_{args.backbone}_ep{args.epochs}'
    
    # wandb 설정
    if args.use_wandb:
        wandb.login(key='')  # your key
        wandb.init(project="brixia")
        wandb.config.update(args)
        experiment_name = model_name
        wandb.run.name = experiment_name
        wandb.run.save()
    
    # 결과 저장 디렉토리 생성
    save_dir = os.path.join(base_path, 'results', model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 데이터 로드
    print('Loading test data...')
    meta_data = pd.read_csv(os.path.join(data_path, 'metadata_global_v2_all.csv'))
    consensus_csv = pd.read_csv(os.path.join(data_path, 'metadata_consensus_v1.csv'))
    
    # 테스트 데이터셋 생성
    test_dataset = BRIXIA(
        filename_list=consensus_csv['Filename'].tolist(),
        label_list=consensus_csv[['ModeA','ModeB','ModeC','ModeD','ModeE','ModeF']].values.tolist(),
        label_df=consensus_csv,
        prefix=os.path.join(data_path, "processed_images/"),
        transform=torch.nn.Sequential(
            transforms.Resize(size=[512, 512], antialias=True),
        ),
        augmentation=None,
        train=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 모델 생성
    print('Creating model...')
    model = Hybrid_e2e_stn(
        args.backbone,
        seg_checkpoint="checkpoint/seg/seg_weights.pt",
        stn_checkpoint="checkpoint/stn/checkpoint.pth"
    )
    
    # 체크포인트 로드
    print(f'Loading checkpoint from {args.checkpoint_path}')
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # 모델 평가
    print('Evaluating model...')
    test_loss, test_acc_list, all_preds, all_labels = evaluate(
        model, 
        test_dataloader, 
        criterion, 
        DEVICE,
        save_stn_results=True,
        save_dir=save_dir
    )
    
    # 결과 저장
    results = {
        'test_loss': test_loss,
        'mean_test_acc': np.mean(test_acc_list),
        'position_acc': test_acc_list
    }
    
    # 결과를 JSON 파일로 저장
    import json
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # wandb에 결과 기록
    if args.use_wandb:
        wandb.log({
            "Test Loss": test_loss,
            "Mean Test ACC": np.mean(test_acc_list),
            **{f"Position {i} ACC": acc for i, acc in enumerate(test_acc_list)}
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test STN-enabled Hybrid Model')
    parser.add_argument('--arch', type=str, default='hybrid_e2e_stn',
                        help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone network')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs the model was trained for')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use wandb for logging')
    
    args = parser.parse_args()
    main(args) 