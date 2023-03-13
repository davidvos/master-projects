import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def poisson_blending(source_img, target_img, mask):
    center = (int(source_img.shape[1]/2), int(source_img.shape[0]/2))
    output = cv.seamlessClone(source_img, target_img, mask, center, cv.NORMAL_CLONE)
    # print(output.dtype)
    return output

def alpha_blending(source_img, target_img, mask):
    mask = mask.astype(float)/255

    foreground = source_img.astype(int)
    background = target_img.astype(int)

    alpha = np.zeros(target_img.shape).astype(float)
    alpha[:,:,0] = mask.squeeze()
    alpha[:,:,1] = mask.squeeze()
    alpha[:,:,2] = mask.squeeze()

    foreground = alpha* foreground
    background = (1.0 - alpha)* background
    output = foreground+ background
    # print(output.dtype)
    return output.astype(np.uint8)

def laplacian_pyramid_blending(source_img, target_img, mask, levels=6):

    # Load the two images of apple and orange
    # Find the Gaussian Pyramids for apple and orange (in this particular example, number of levels is 6)
    # From Gaussian Pyramids, find their Laplacian Pyramids
    # Now join the left half of apple and right half of orange in each levels of Laplacian Pyramids
    # Finally from this joint image pyramids, reconstruct the original image.
    
    # generate Gaussian pyramid for source and target
    G_source = source_img.copy()
    G_target = target_img.copy()
    G_mask   = (mask/255).copy()

    gp_source = [G_source]
    gp_target = [G_target]
    gp_mask   = [G_mask.squeeze()]

    for i in range(levels):
        G_source = cv.pyrDown(G_source)
        G_target = cv.pyrDown(G_target)
        G_mask   = cv.pyrDown(G_mask.squeeze())
        gp_source.append(G_source)
        gp_target.append(G_target)
        gp_mask.append(G_mask)

    # generate Laplacian Pyramid for source and target
    lp_source = [gp_source[levels-1]]
    lp_target = [gp_target[levels-1]]
    gp_mask_reverse = [gp_mask[levels-1]]

    for i in range(levels-1,0,-1):
        GE_source = cv.pyrUp(gp_source[i])
        GE_target = cv.pyrUp(gp_target[i])
        L_source = cv.subtract(gp_source[i-1],GE_source)
        L_target = cv.subtract(gp_target[i-1],GE_target)
        # misschien laatste gp_mask fout

        lp_source.append(L_source)
        lp_target.append(L_target)
        gp_mask_reverse.append(gp_mask[i-1])

    # Now add left and right halves of images in each level
    Total = []
    for index, (l_source, l_target) in enumerate(zip(lp_source,lp_target)):
        alpha = np.zeros(l_source.shape).astype(float)
        alpha[:,:,0] = gp_mask_reverse[index].squeeze()
        alpha[:,:,1] = gp_mask_reverse[index].squeeze()
        alpha[:,:,2] = gp_mask_reverse[index].squeeze()
        ls = l_source * alpha + l_target * (1.0 - alpha)

        Total.append(ls)
    
    final = Total[0]
    for i in range(1,levels):
        final = cv.pyrUp(final)
        final = cv.add(final, Total[i])
    final = final/ final.max()* 255
    # print(final.max())
    # print(final.min())
    # plt.imshow(final.astype(int)[...,::-1] )
    # plt.show()
    return final.astype(np.uint8)