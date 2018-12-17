import numpy as np
import cv2
from scipy import ndimage
import time
from os.path import exists

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def census(patch1, patch2):
    y_size = patch1.shape[0]
    x_size = patch1.shape[1]

    zero_patch = np.zeros(patch1.shape)
    p1_cent = patch1[int(y_size / 2), int(x_size / 2)]
    p2_cent = patch2[int(y_size / 2), int(x_size / 2)]
    patch1 -= p1_cent
    patch2 -= p2_cent
    patch1 = np.maximum(patch1, zero_patch)
    patch1 = patch1 / (patch1 + 10e-8)
    patch2 = np.maximum(patch2, zero_patch)
    patch2 = patch2 / (patch2 + 10e-8)
    sum_patch = (patch1 + patch2).astype(np.int16)
    xor_patch = sum_patch[np.where(sum_patch == 1)]
    cost = len(xor_patch)
    return cost

def normalize(patch):
    nor_patch = ((patch - np.mean(patch)) / np.std(patch) + 10e-8)
    return nor_patch

def computeDisp(Il, Ir, scale_factor, max_disp):

    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    block_size = 9
    half_wd = int(block_size / 2)

    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir

    Il = ndimage.median_filter(Il, 3)
    Ir = ndimage.median_filter(Ir, 3)
    cost_volumn = np.zeros((h, w, max_disp+1), dtype=np.float32)

    #Il = cv2.bilateralFilter(Il, 5, 21, 21)
    #Ir = cv2.bilateralFilter(Ir, 5, 21, 21)

    for y in range(h):
        print (y)
        for x in range(w):
            top   = max(0, y - half_wd)
            left  = max(0, x - half_wd)
            bot   = min(h, y + half_wd)
            right = min(w, x + half_wd)

            l_patch = Il[top:bot, left:right]
            r_patch = Ir[top:bot, left:right]
            l_patch_g = cv2.cvtColor(l_patch, cv2.COLOR_BGR2GRAY)#.astype(np.float32)
            r_patch_g = cv2.cvtColor(r_patch, cv2.COLOR_BGR2GRAY)#.astype(np.float32)

            dis = list()
            for shift in range(-max_disp, 1):
                leftend = max(0, left+shift)
                rightend = min(w, right+shift)
                x_interval = rightend - leftend

                if x_interval > 0:
                    r_patch = Ir[top:bot, leftend:rightend]
                    l_patch_2 = l_patch[:, :x_interval]

                    #nor_l_patch = normalize(l_patch_2)
                    #nor_r_patch = normalize(r_patch)
                    l_patch_g = cv2.cvtColor(l_patch_2, cv2.COLOR_BGR2GRAY)#.astype(np.float32)
                    r_patch_g = cv2.cvtColor(r_patch, cv2.COLOR_BGR2GRAY)#.astype(np.float32)
                    #dis.append(census(l_patch_g, r_patch_g))
                    cost_volumn[y, x, -shift] = correlation_coefficient(l_patch_g, r_patch_g)
                    #dis.append(-correlation_coefficient(l_patch_g, r_patch_g))
            #dis = np.array(dis)

            #labels[y, x] = (max_disp - np.argmin(dis)) * scale_factor

    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()

    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering

    for z in range(max_disp+1):
        cost_volumn[:, :, z] = cv2.bilateralFilter(cost_volumn[:, :, z], 5, 21, 21)
    labels = np.argmax(cost_volumn, axis=2) * scale_factor
    #tmp
    for i in range(max_disp):
        labels[:, i] = labels[:, max_disp+1]
    #labels = ndimage.median_filter(labels, 3)
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))
    return labels


def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, scale_factor, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels))

    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, scale_factor, max_disp)# / 2
    cv2.imwrite('venus.png', np.uint8(labels))

    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, scale_factor, max_disp)# / 4
    cv2.imwrite('teddy.png', np.uint8(labels))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, scale_factor, max_disp)# / 4
    cv2.imwrite('cones.png', np.uint8(labels))


if __name__ == '__main__':
    main()
