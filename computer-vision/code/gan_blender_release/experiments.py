import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as F
import imageio
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

import res_unet
import utils
from SwappedDataset import SwappedDatasetLoader
from img_utils import rgb2tensor, tensor2rgb
import img_utils
from blending_utils import poisson_blending, alpha_blending, laplacian_pyramid_blending

torch.backends.cudnn.benchmark = True

cudaDevice = ''

if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
            'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
        cudaDevice)

def transfer_mask(img1, img2, mask):
    return img1 * mask + img2 * (1 - mask)

def blend_imgs_bgr(source_img, target_img, mask, blending_technique):
    # Implement poisson blending here. You can us the built-in seamlessclone
    # function in opencv which is an implementation of Poisson Blending.
    if blending_technique == 'poisson':
        img =  poisson_blending(source_img, target_img, mask)

    if blending_technique == 'alpha':
        img =  alpha_blending(source_img, target_img, mask)

    if blending_technique == "laplacian":
        img = laplacian_pyramid_blending(source_img, target_img, mask)
    
    if not img.any():
        raise ValueError(f"Passed blending technique: \"{blending_technique}\" is invalid")

    return img

def blend_imgs(source_tensor, target_tensor, mask_tensor, blending_technique):
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
        mask = np.round(mask * 255).astype('uint8')
        out_bgr = blend_imgs_bgr(source_img, target_img, mask, blending_technique)
        out_tensors.append(img_utils.bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)

def normalize(img):
    if len(img.shape)> 3:
        for i in range(img.shape[0]):
            if not i:
                all_normalized = normalize(img[i,:,:,:]).unsqueeze(0)
            else:
                all_normalized = torch.cat((all_normalized, normalize(img[i,:,:,:]).unsqueeze(0)))
        # print(all_normalized.shape)
        # quit()
        # return torch.Tensor([normalize(o) for o in img])
    else:
        all_normalized = F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return all_normalized

def blend_test_dataset(checkpoint_path, save_path, blending_technique):
    with torch.no_grad():
        test_list = '../../data_set/test.str'
        data_root='../../data_set/data/'
        img_resolution = 256

        testSet = SwappedDatasetLoader(test_list, data_root, img_resolution)
        testLoader = DataLoader(testSet, batch_size=2, shuffle=False, num_workers=0,
                                        pin_memory=True, drop_last=True)

        G = res_unet.MultiScaleResUNet(in_nc=7)
        G, _, _ = utils.loadModels(G, checkpoint_path, optims=None, Test=False, device=device)
        G = G.to(device)
        G.eval()    
        pbar = tqdm(enumerate(testLoader), total=len(testLoader), leave=False)
            
        for i, data in pbar:

            _, images = data
            
            # Feed the network with images from test set
            target = normalize(images['target']).to(device)
            swap = normalize(images['swap']).to(device)
            swap_mask = images['mask'].to(device)

            target_hat = transfer_mask(swap, target, swap_mask)
            model_input = torch.cat((target_hat, target, swap_mask), 1)
            pred = G(model_input)

            for index, b in enumerate(range(pred.shape[0])):
                grid = tensor2rgb(pred[b].detach())
                imageio.imwrite(save_path + '/' + blending_technique + str(i) + '-' + str(index) + '.png', grid)

def blend_optimized_results(checkpoint_path, save_path, blending_technique):

    G = res_unet.MultiScaleResUNet(in_nc=7)
    G, _, _ = utils.loadModels(G, checkpoint_path, optims=None, Test=False, device=device)
    G = G.to(device)
    G.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
    ])

    directory = "Optimized_dataset/"

    for i in range(5):
        swap = transform(rgb2tensor(np.load(directory + 'swap_' + str(i) + '.npy'))).to(device)
        target = transform(rgb2tensor(np.load(directory + 'target_' + str(i) + '.npy'))).to(device)
        swap_mask = (torch.mean(swap, 1, keepdim=True)>-1).float().to(device)

        target_hat = transfer_mask(swap, target, swap_mask)
        model_input = torch.cat((target_hat, target, swap_mask), 1).float()
        pred = G(model_input)

        for index, b in enumerate(range(pred.shape[0])):
            grid = tensor2rgb(pred[b].detach())
            imageio.imwrite(save_path + '/' + blending_technique + str(i) + '-' + str(index) + '.png', grid)

def blend_naive_results(checkpoint_path, save_path, blending_technique):

    G = res_unet.MultiScaleResUNet(in_nc=7)
    G, _, _ = utils.loadModels(G, checkpoint_path, optims=None, Test=False, device=device)
    G = G.to(device)
    G.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    directory = "Naive_dataset/"

    for i in range(1, 6):
        swap = normalize(transform(Image.open(directory + str(i) + '_swap.png').convert('RGB'))).unsqueeze(0).to(device)
        target = normalize(transform(Image.open(directory + str(i) + '_target.png').convert('RGB'))).unsqueeze(0).to(device)
        swap_mask = (torch.mean(swap, 1, keepdim=True)>-1).float().to(device)
        target_hat = transfer_mask(swap, target, swap_mask)
        model_input = torch.cat((target_hat, target, swap_mask), 1).float()
        pred = G(model_input)

        for index, b in enumerate(range(pred.shape[0])):
            image = tensor2rgb(pred[b].detach())
            imageio.imwrite(save_path + '/' + blending_technique + str(i) + '-' + str(index) + '.png', image)

def create_ground_truths(save_path, blending_technique):
    test_list = '../../data_set/test.str'
    data_root='../../data_set/data/'
    img_resolution = 256

    testSet = SwappedDatasetLoader(test_list, data_root, img_resolution)
    testLoader = DataLoader(testSet, batch_size=2, shuffle=False, num_workers=0,
                                    pin_memory=True, drop_last=True)

    pbar = tqdm(enumerate(testLoader), total=len(testLoader), leave=False)
        
    for i, data in pbar:

        _, images = data
        
        # Feed the network with images from test set
        target = normalize(images['target']).to(device)
        swap = normalize(images['swap']).to(device)
        swap_mask = images['mask'].to(device)

        target_hat = transfer_mask(swap, target, swap_mask)

        ground_truth = blend_imgs(target_hat, target, swap_mask, blending_technique).to(device)

        for index, b in enumerate(range(ground_truth.shape[0])):
            image = tensor2rgb(ground_truth[b].detach())
            imageio.imwrite(save_path + '/' + blending_technique + str(i) + '-' + str(index) + '.png', image)

def blend_dl_results(checkpoint_path, save_path, blending_technique):
    G = res_unet.MultiScaleResUNet(in_nc=7)
    G, _, _ = utils.loadModels(G, checkpoint_path, optims=None, Test=False, device=device)
    G = G.to(device)
    G.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    directory = "DL_Dataset/"

    for i in range(1, 6):
        swap = normalize(transform(Image.open(directory + str(i) + '_swap.png').convert('RGB'))).unsqueeze(0).to(device)
        target = normalize(transform(Image.open(directory + str(i) + '_target.png').convert('RGB'))).unsqueeze(0).to(device)
        swap_mask = (torch.mean(swap, 1, keepdim=True)>-1).float().to(device)
        target_hat = transfer_mask(swap, target, swap_mask)
        model_input = torch.cat((target_hat, target, swap_mask), 1).float()
        pred = G(model_input)

        for index, b in enumerate(range(pred.shape[0])):
            image = tensor2rgb(pred[b].detach())
            imageio.imwrite(save_path + '/' + blending_technique + str(i) + '-' + str(index) + '.png', image)

def create_original_dataset(save_path):
    file_names = open('../../data_set/test.str', 'r')
    file_list = file_names.readlines()

    for file in file_list:
        splitted_filename = file.strip().split('_')
        x = splitted_filename[0]
        y = splitted_filename[2]
        z = splitted_filename[3].split('.')[0]
        
        shutil.copy('../../data_set/data/' + x + '_fg_' + z + '.png', save_path)
        shutil.copy('../../data_set/data/' + x + '_bg_' + y + '.png', save_path)

if __name__ == '__main__':

    os.makedirs('Generated_datasets/original', exist_ok=True)

    os.makedirs('Generated_datasets/GroundTruths/poisson', exist_ok=True)
    os.makedirs('Generated_datasets/GroundTruths/alpha', exist_ok=True)
    os.makedirs('Generated_datasets/GroundTruths/laplacian', exist_ok=True)

    os.makedirs('Generated_datasets/TestSet7/Rec1/poisson', exist_ok=True)
    os.makedirs('Generated_datasets/TestSet7/Rec1/alpha', exist_ok=True)
    os.makedirs('Generated_datasets/TestSet7/Rec1/laplacian', exist_ok=True)

    os.makedirs('Generated_datasets/TestSet15/Rec1/poisson', exist_ok=True)
    os.makedirs('Generated_datasets/TestSet15/Rec1/alpha', exist_ok=True)
    os.makedirs('Generated_datasets/TestSet15/Rec1/laplacian', exist_ok=True)

    os.makedirs('Generated_datasets/TestSet7/Rec05/poisson', exist_ok=True)
    os.makedirs('Generated_datasets/TestSet7/Rec05/alpha', exist_ok=True)
    os.makedirs('Generated_datasets/TestSet7/Rec05/laplacian', exist_ok=True)

    os.makedirs('Generated_datasets/TestSet15/Rec05/poisson', exist_ok=True)
    os.makedirs('Generated_datasets/TestSet15/Rec05/alpha', exist_ok=True)
    os.makedirs('Generated_datasets/TestSet15/Rec05/laplacian', exist_ok=True)

    os.makedirs('Generated_datasets/OptimizedSet', exist_ok=True)
    os.makedirs('Generated_datasets/Naive', exist_ok=True)
    os.makedirs('Generated_datasets/DL/poisson', exist_ok=True)

    create_original_dataset('Generated_datasets/original')

    create_ground_truths('Generated_datasets/GroundTruths/poisson/', 'poisson')
    create_ground_truths('Generated_datasets/GroundTruths/alpha/', 'alpha')
    create_ground_truths('Generated_datasets/GroundTruths/laplacian/', 'laplacian')
    
    blend_test_dataset('Exp_blender/checkpoints/poisson/checkpoint_G_15_05.pth', 'Generated_datasets/TestSet15/Rec05/poisson/', 'poisson')
    blend_test_dataset('Exp_blender/checkpoints/alpha/checkpoint_G_15_05.pth', 'Generated_datasets/TestSet15/Rec05/alpha/', 'alpha')
    blend_test_dataset('Exp_blender/checkpoints/laplacian/checkpoint_G_15_05.pth', 'Generated_datasets/TestSet15/Rec05/laplacian/', 'laplacian')

    blend_test_dataset('Exp_blender/checkpoints/poisson/checkpoint_G_7_05.pth', 'Generated_datasets/TestSet7/Rec05/poisson/', 'poisson')
    blend_test_dataset('Exp_blender/checkpoints/alpha/checkpoint_G_7.pth_05', 'Generated_datasets/TestSet7/Rec05/alpha/', 'alpha')
    blend_test_dataset('Exp_blender/checkpoints/laplacian/checkpoint_G_7_05.pth', 'Generated_datasets/TestSet7/Rec05/laplacian/', 'laplacian')

    blend_test_dataset('Exp_blender/checkpoints/poisson/checkpoint_G_15_1.pth', 'Generated_datasets/TestSet15/Rec1/poisson/', 'poisson')
    blend_test_dataset('Exp_blender/checkpoints/alpha/checkpoint_G_15_1.pth', 'Generated_datasets/TestSet15/Rec1/alpha/', 'alpha')
    blend_test_dataset('Exp_blender/checkpoints/laplacian/checkpoint_G_15_1.pth', 'Generated_datasets/TestSet15/Rec1/laplacian/', 'laplacian')

    blend_test_dataset('Exp_blender/checkpoints/poisson/checkpoint_G_7_1.pth', 'Generated_datasets/TestSet7/Rec1/poisson/', 'poisson')
    blend_test_dataset('Exp_blender/checkpoints/alpha/checkpoint_G_7_1.pth', 'Generated_datasets/TestSet7/Rec1/alpha/', 'alpha')
    blend_test_dataset('Exp_blender/checkpoints/laplacian/checkpoint_G_7_1.pth', 'Generated_datasets/TestSet7/Rec1/laplacian/', 'laplacian')
    
    blend_optimized_results('Exp_blender/checkpoints/poisson/checkpoint_G_7_05.pth', 'Generated_datasets/OptimizedSet/poisson/', 'poisson')
    blend_naive_results('Exp_blender/checkpoints/poisson/checkpoint_G_7_05.pth', 'Generated_datasets/Naive/poisson/', 'poisson')
    blend_dl_results('Exp_blender/checkpoints/poisson/checkpoint_G_7_05.pth', 'Generated_datasets/DL/poisson/', 'poisson')
