import os
import time
from numpy.lib.utils import source
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


import vgg_loss
import discriminators_pix2pix
import res_unet
import gan_loss
from SwappedDataset import SwappedDatasetLoader
import utils
import img_utils

from blending_utils import poisson_blending, alpha_blending, laplacian_pyramid_blending


# Configurations
######################################################################
# Fill in your experiment names and the other required components
experiment_name = 'laplacian'
data_root = '../../data_set/data/'
train_list = '../../data_set/train.str'
test_list = '../../data_set/test.str'
batch_size = 2
nthreads = 0
max_epochs = 20
displayIter = 20
saveIter = 1
img_resolution = 256

lr_gen = 1e-4
lr_dis = 1e-4

momentum = 0.9
weightDecay = 1e-4
step_size = 30
gamma = 0.1

pix_weight = 0.1
rec_weight = 1.0
gan_weight = 0.001
######################################################################
# Independent code. Don't change after this line. All values are automatically
# handled based on the configuration part.

if batch_size < nthreads:
    nthreads = batch_size
check_point_loc = 'Exp_blender/checkpoints/' #% experiment_name.replace(' ', '_')
visuals_loc = 'Exp_blender/visuals/'# % experiment_name.replace(' ', '_')
os.makedirs(check_point_loc, exist_ok=True)
os.makedirs(visuals_loc, exist_ok=True)
checkpoint_pattern = check_point_loc + 'checkpoint_%s_%d.pth'
logTrain = check_point_loc + 'LogTrain.txt'

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

done = u'\u2713'

print('[I] STATUS: Initiate Network and transfer to device...', end='')

G = res_unet.MultiScaleResUNet(in_nc=7)
D = discriminators_pix2pix.MultiscaleDiscriminator(use_sigmoid=True)

print(done)

print('[I] STATUS: Load Networks...', end='')
# Load your pretrained models here. Pytorch requires you to define the model
# before loading the weights, since the weight files does not contain the model
# definition. Make sure you transfer them to the proper training device. Hint:
    # use the .to(device) function, where device is automatically detected
    # above.

G, _, _ = utils.loadModels(G, check_point_loc + 'checkpoint_G.pth', optims=None, Test=False, device=device)
D, _, _ = utils.loadModels(D, check_point_loc + 'checkpoint_D.pth', optims=None, Test=False, device=device)
G = G.to(device)
D = D.to(device)

print(done)

print('[I] STATUS: Initiate optimizer...', end='')

# Define your optimizers and the schedulers and connect the networks from
# before
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_gen, betas=(momentum, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_dis, betas=(momentum, 0.999))
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=step_size,gamma=gamma)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=step_size,gamma=gamma)


print(done)

print('[I] STATUS: Initiate Criterions and transfer to device...', end='')

# Define your criterions here and transfer to the training device. They need to
# be on the same device type.
gan_loss = gan_loss.GANLoss().to(device)
l1_loss = nn.L1Loss().to(device)
vgg_loss = vgg_loss.VGGLoss().to(device)

print(done)

print('[I] STATUS: Initiate Dataloaders...')

trainSet = SwappedDatasetLoader(train_list, data_root, img_resolution)
trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=nthreads,
                            pin_memory=True, drop_last=True)

testSet = SwappedDatasetLoader(test_list, data_root, img_resolution)
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=nthreads,
                            pin_memory=True, drop_last=True)

print(done)

print('[I] STATUS: Initiate Logs...', end='')
trainLogger = open(logTrain, 'w')
print(done)


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
    # print("img max: ",img.max())
    # print("img dtype: ",img.dtype)
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

# def 

def Train(G, D, epoch_count, iter_count, blending_technique):
    G.train(True)
    D.train(True)
    epoch_count += 1
    # print('we gaan traine!!!')
    pbar = tqdm(enumerate(trainLoader), total=len(trainLoader), leave=False)
    
    Epoch_time = time.time()

    for i, data in pbar:
        iter_count += 1

        _, images = data

        # 1) Load and transfer data to device
        target = normalize(images['target']).to(device)
        swap = normalize(images['swap']).to(device)
        swap_mask = images['mask'].to(device)
        
        target_hat = transfer_mask(swap, target, swap_mask)
        
        # ground_truth = blend_imgs(target_hat, target, swap_mask, experiment_name).to(device)
        ground_truth = blend_imgs(target_hat, target, swap_mask, blending_technique).to(device)
        model_input = torch.cat((target_hat, target, swap_mask), 1)

        # target = target.cpu() 
        swap = swap.cpu()
        swap_mask = swap_mask.cpu()
        # quit()
        # 2) Feed the data to the networks. 
        # 4) Calculate the losses.
        # 5) Perform backward calculation.
        # 6) Perform the optimizer step.

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        # torch.autograd.set_detect_anomaly(True)

        generator_out = G(model_input)

        # discriminator part
        fake_discriminator_out = D(generator_out.detach())
        real_discriminator_out = D(target)

        fake_discriminator_loss = 0.5 * gan_loss(fake_discriminator_out, False)
        real_discriminator_loss = 0.5 * gan_loss(real_discriminator_out, True)
        total_discriminator_loss = fake_discriminator_loss + real_discriminator_loss
        # del fake_discriminator_out
        # del real_discriminator_out
        # generator part

        lid = vgg_loss(generator_out, ground_truth)
        lpixel = l1_loss(generator_out, ground_truth)
        lrec = 0.1 * lpixel + 0.5 * lid

        # ld = gan_loss(real_discriminator_out, True) + gan_loss(fake_discriminator_out, False)
        disc_out = D(generator_out)
        
        lg = gan_loss(disc_out, True)
        # lg2 = -(fake_discriminator_loss.detach() + real_discriminator_loss.detach())

        # generator_loss = lrec
        generator_loss =  + 0.001 * lg #+ 0.001 * lg2

        generator_loss.backward()
        optimizer_G.step()

        total_discriminator_loss.backward()
        # real_discriminator_loss.backward()

        optimizer_D.step()




        if iter_count % displayIter == 0:
            # Write to the log file.
            trainLogger.write(f"discriminator loss: {fake_discriminator_loss + real_discriminator_loss} generator loss: {generator_loss}")


        if iter_count == 10:
            break
        # Print out the losses here. Tqdm uses that to automatically print it
        # in front of the progress bar.
        pbar.set_description()

    # Save output of the network at the end of each epoch. The Generator

    t_source, t_swap, t_target, t_pred, t_blend = Test(G, blending_technique)
    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())
        imageio.imwrite(visuals_loc + '/Epoch_%d_output_%d.png' %
                        (epoch_count, b), grid)

    utils.saveModels(G, optimizer_G, iter_count,
                     checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, optimizer_D, iter_count,
                     checkpoint_pattern % ('D', epoch_count))
    tqdm.write('[!] Model Saved!')

    # return np.nanmean(total_loss_pix),\
    #     np.nanmean(total_loss_id), np.nanmean(total_loss_attr),\
    #     np.nanmean(total_loss_rec), np.nanmean(total_loss_G_Gan),\
    #     np.nanmean(total_loss_D_Gan), iter_count


def Test(G, blending_technique):
    with torch.no_grad():
        G.eval()
        t = enumerate(testLoader)
        i, images = next(t)
        images = images[1]

        # Feed the network with images from test set
        source = normalize(images['source']).to(device)
        target = normalize(images['target']).to(device)
        swap = normalize(images['swap']).to(device)
        swap_mask = images['mask'].to(device)

        target_hat = transfer_mask(swap, target, swap_mask)
        
        ground_truth = blend_imgs(target_hat, target, swap_mask, blending_technique)

        model_input = torch.cat((target_hat, target, swap_mask), 1)
    
        # Blend images
        pred = G(model_input)

        return source, swap, target, pred, ground_truth

if __name__ == '__main__':
    iter_count = 0
    # Print out the experiment configurations. You can also save these to a file if
    # you want them to be persistent.
    print('[*] Beginning Training:')
    print('\tMax Epoch: ', max_epochs)
    print('\tLogging iter: ', displayIter)
    print('\tSaving frequency (per epoch): ', saveIter)
    print('\tModels Dumped at: ', check_point_loc)
    print('\tVisuals Dumped at: ', visuals_loc)
    print('\tExperiment Name: ', experiment_name)

    for i in range(max_epochs):
        Train(G, D, i, iter_count, 'laplacian')

        scheduler_D.step()
        scheduler_G.step()
        # Call the Train function here
        # Step through the schedulers if using them.
        # You can also print out the losses of the network here to keep track of
        # epoch wise loss.

    trainLogger.close()