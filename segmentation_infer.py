import numpy as np
import cv2 

from segementation.HybridGNet2IGSC import HybridGNet
from seg_utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart
import scipy.sparse as sp
import torch
import pandas as pd
from zipfile import ZipFile
from glob import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hybrid = None

def getDenseMask(landmarks, h, w):
    
    RL = landmarks[0:44]
    LL = landmarks[44:94]
    H = landmarks[94:]
    
    img = np.zeros([h, w], dtype = 'uint8')
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    H = H.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 1, -1)
    img = cv2.drawContours(img, [H], -1, 2, -1)
    
    return img

def getMasks(landmarks, h, w):
    
    RL = landmarks[0:44]
    LL = landmarks[44:94]
    H = landmarks[94:]
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    H = H.reshape(-1, 1, 2).astype('int')
    
    RL_mask = np.zeros([h, w], dtype = 'uint8')
    LL_mask = np.zeros([h, w], dtype = 'uint8')
    H_mask = np.zeros([h, w], dtype = 'uint8')
    
    RL_mask = cv2.drawContours(RL_mask, [RL], -1, 255, -1)
    LL_mask = cv2.drawContours(LL_mask, [LL], -1, 255, -1)
    H_mask = cv2.drawContours(H_mask, [H], -1, 255, -1)

    return RL_mask, LL_mask, H_mask

def drawOnTop(img, landmarks, original_shape):
    h, w = original_shape
    output = getDenseMask(landmarks, h, w)
    
    image = np.zeros([h, w, 3])
    image[:,:,0] = img + 0.3 * (output == 1).astype('float') - 0.1 * (output == 2).astype('float')
    image[:,:,1] = img + 0.3 * (output == 2).astype('float') - 0.1 * (output == 1).astype('float') 
    image[:,:,2] = img - 0.1 * (output == 1).astype('float') - 0.2 * (output == 2).astype('float') 

    image = np.clip(image, 0, 1)
    
    RL, LL, H = landmarks[0:44], landmarks[44:94], landmarks[94:]
    
    # Draw the landmarks as dots
    
    for l in RL:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 0, 1), -1)
    for l in LL:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 0, 1), -1)
    for l in H:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 1, 0), -1)
    
    return image
    

def loadModel(device):    
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
    
    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_))

    config['latents'] = 64
    config['inputsize'] = 1024

    f = 32
    config['filters'] = [2, f, f, f, f//2, f//2, f//2]
    config['skip_features'] = f

    hybrid = HybridGNet(config.copy(), D_t, U_t, A_t).to(device)
    hybrid.load_state_dict(torch.load("checkpoint/seg/seg_weights.pt", map_location=torch.device(device)))
    hybrid.eval()
    print(hybrid)
    
    return hybrid


def pad_to_square(img):
    h, w = img.shape[:2]
    
    if h > w:
        padw = (h - w) 
        auxw = padw % 2
        img = np.pad(img, ((0, 0), (padw//2, padw//2 + auxw)), 'constant')
        
        padh = 0
        auxh = 0
        
    else:
        padh = (w - h) 
        auxh = padh % 2
        img = np.pad(img, ((padh//2, padh//2 + auxh), (0, 0)), 'constant')

        padw = 0
        auxw = 0
        
    return img, (padh, padw, auxh, auxw)
    

def preprocess_batch(input_imgs):
    batch = []
    infos = []
    for input_img in input_imgs:
        img, padding = pad_to_square(input_img)
        h, w = img.shape[:2]
        if h != 1024 or w != 1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        batch.append(img)
        infos.append((h, w, padding))
    return np.stack(batch), infos


def removePreprocess_batch(outputs, infos):
    processed_outputs = []
    for output, info in zip(outputs, infos):
        h, w, padding = info
        if h != 1024 or w != 1024:
            output = output * h
        else:
            output = output * 1024
        padh, padw, auxh, auxw = padding
        output[:, 0] = output[:, 0] - padw // 2
        output[:, 1] = output[:, 1] - padh // 2
        processed_outputs.append(output)
    return np.stack(processed_outputs)   


def zip_files(files):
    with ZipFile("complete_results.zip", "w") as zipObj:
        for idx, file in enumerate(files):
            zipObj.write(file, arcname=file.split("/")[-1])
    return "complete_results.zip"


def segment_batch(input_imgs,save_folder):
    global hybrid, device

    if hybrid is None:
        hybrid = loadModel(device)

    input_img_mats = [cv2.imread(input_img, 0)/255.0 for input_img in input_imgs]
    
    
    imgs, infos = preprocess_batch(input_img_mats)
    data = torch.from_numpy(imgs).unsqueeze(1).to(device).float()

    with torch.no_grad():
        outputs=hybrid(data)[0] # x,pos1,pos2
        outputs = outputs.cpu().numpy()
        # (7,120,2) ->[(120,2)... 7개]. numpy to list of numpy
        output_list = np.split(outputs, len(outputs), axis=0)
        output_list = [output[0] for output in output_list]
        #import pdb;pdb.set_trace()
        #print()
    

    outputs = removePreprocess_batch(outputs, infos)
    

    outputs = outputs.astype('int')
    outsegs = []

    for i, input_img in enumerate(input_img_mats):
        original_shape = input_img.shape[:2]
        #outseg = drawOnTop(input_img, outputs[i], original_shape)
        #outsegs.append(outseg)

        #seg_to_save = (outseg.copy() * 255).astype('uint8')
        #overlap_folder = "/hdd/project/cxr_haziness/data/0407_0608/gnn_mask/left/"
        #cv2.imwrite(f"{overlap_folder}{i}", cv2.cvtColor(seg_to_save, cv2.COLOR_RGB2BGR))

        RL = outputs[i][:44]
        LL = outputs[i][44:94]
        H = outputs[i][94:]

        # np.savetxt(f"tmp/RL_landmarks_{i}.txt", RL, delimiter=" ", fmt="%d")
        # np.savetxt(f"tmp/LL_landmarks_{i}.txt", LL, delimiter=" ", fmt="%d")
        # np.savetxt(f"tmp/H_landmarks_{i}.txt", H, delimiter=" ", fmt="%d")

        RL_mask, LL_mask, H_mask = getMasks(outputs[i], original_shape[0], original_shape[1])

        all_mask = RL_mask + LL_mask
        
        # 파일 이름 추출
        file_name = input_imgs[i].split("/")[-1]
        cv2.imwrite(f"{save_folder}{file_name}", all_mask)
        #
        
                

    return outsegs

if __name__ == "__main__":    
    
    hybrid = loadModel(device)
    
    #input_img_folder = "/hdd/project/cxr_haziness/data/0407_0608/processed_images/"
    #save_folder = "/hdd/project/cxr_haziness/data/0407_0608/gnn_mask/all/"
    
    # MIMIC
    #input_img_folder = "/hdd/project/cxr_haziness/data/mimic/haziness/"
    #save_folder = "/hdd/project/cxr_haziness/data/mimic/haziness_gnn_mask/"
    
    # RSNA
    #input_img_folder = "/hdd/project/cxr_haziness/data/stage_1_train_pngs/preprocess_label1/"
    # input_img_folder = "/hdd/project/cxr_haziness/data/stage_1_train_pngs/preprocess_images/"
    # save_folder = "/hdd/project/cxr_haziness/data/stage_1_train_pngs/gnn_mask_label1/"
    
    # RALO
    # input_img_folder = "/hdd/project/cxr_haziness/data/RALO/CXR_images_scored/"
    # save_folder = "/hdd/project/cxr_haziness/data/RALO/gnn_mask/"    
    
    # private update
    # input_img_folder = "/hdd/project/cxr_haziness/data/CXR_23_1113/processed_images/"
    # save_folder = "/hdd/project/cxr_haziness/data/CXR_23_1113/gnn_mask/all/"
    
    
    # brixia
    # input_img_folder = "/hdd/project/cxr_haziness/data/brixia/processed_images/"
    # save_folder = "/hdd/project/cxr_haziness/data/brixia/processed_masks/"    
    
    #Edema
    #input_img_folder = "/hdd/project/cylce_het/dataset/edema_raw_images/"
    #save_folder = "/hdd/project/cylce_het/dataset/edema_mask/"

    #inha
    input_img_folder = "/hdd/project/cylce_het/dataset/inha_raw_images/"
    save_folder = "/hdd/project/cylce_het/dataset/inha_mask/"    
    
    
    img_list = glob(input_img_folder+"*.png")
    print(len(img_list))
    
    # batch size =16으로 데이터들을 묶어서 처리한다.
    for i in range(0,len(img_list),16):
        image_output = segment_batch(img_list[i:i+16],save_folder)
        print(i," batch ",len(image_output))
    #image_output = segment_batch(img_list,save_folder)
    
    
    #print(results)
    
    
    