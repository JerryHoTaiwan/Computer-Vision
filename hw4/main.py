import numpy as np
import cv2
from scipy import ndimage
import time
from os.path import exists
import random
from ref import BSM
import psutil

np.random.seed(666)
BLACK = [0, 0, 0]

class BinaryCost():
    def __init__(self, img1, img2, pair, half_wd=13, num_pair=4096):
        self.img1 = img1
        self.img2 = img2
        self.num_pair = num_pair

        self.first_y = pair[0]
        self.first_x = pair[1]
        self.second_y = pair[2]
        self.second_x = pair[3]
        self.half_wd = half_wd

    def set_bits(self):
        h, w = self.img1.shape
        self.left_bits = np.zeros((h, w, self.num_pair), dtype=bool)
        self.right_bits = np.zeros((h, w, self.num_pair), dtype=bool)

        for y in range(self.half_wd, h-self.half_wd):
            print (y)

            for x in range(self.half_wd, w-self.half_wd):
                left_bits, right_bits = self.get_bits(y, x, 0)
                self.left_bits[y-self.half_wd, x-self.half_wd] = left_bits
                self.right_bits[y-self.half_wd, x-self.half_wd] = right_bits
                del left_bits, right_bits

    def get_bits(self, y_pos, x_pos, sft):
        self.select_patch_11 = self.img1[y_pos + self.first_y, x_pos + self.first_x]
        self.select_patch_12 = self.img1[y_pos + self.second_y, x_pos + self.second_x]

        self.select_patch_21 = self.img2[y_pos + self.first_y, x_pos - sft + self.first_x]
        self.select_patch_22 = self.img2[y_pos + self.second_y, x_pos - sft + self.second_x]

        result_patch_1 = self.select_patch_11 > self.select_patch_12
        result_patch_2 = self.select_patch_21 > self.select_patch_22
        del self.select_patch_11, self.select_patch_12, self.select_patch_21, self.select_patch_22

        return result_patch_1, result_patch_2

def get_random_pair(num_pair=4096):

    pair_seq = (np.random.normal(0., 4.0, num_pair*4))#.astype(np.int16)
    pair_seq = np.clip(pair_seq, -13, 13).astype(np.int_)

    first_y = pair_seq[:num_pair]    
    first_x = pair_seq[num_pair:num_pair*2]    
    second_y = pair_seq[num_pair*2:num_pair*3]    
    second_x = pair_seq[num_pair*3:]

    return (first_y, first_x, second_y, second_x)

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def normalize(patch):
    nor_patch = ((patch - np.mean(patch)) / np.std(patch) + 10e-8)
    return nor_patch

def computeDisp(Il, Ir, scale_factor, max_disp):

    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    block_size = 26
    half_wd = int(block_size / 2)

    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir

    #Il = ndimage.median_filter(Il, 3)
    #Ir = ndimage.median_filter(Ir, 3)
    cost_volume = np.zeros((h, w, max_disp+1), dtype=np.float32)
    cost_volume += 10e8

    Il = cv2.bilateralFilter(Il, 5, 21, 21)
    Ir = cv2.bilateralFilter(Ir, 5, 21, 21)
    Il = cv2.copyMakeBorder(Il, half_wd, half_wd, half_wd, half_wd, cv2.BORDER_CONSTANT, value=BLACK)
    Ir = cv2.copyMakeBorder(Ir, half_wd, half_wd, half_wd, half_wd, cv2.BORDER_CONSTANT, value=BLACK)
    Il = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    pairs = get_random_pair()

    bincost = BinaryCost(img1=Il, img2=Ir, pair=pairs, half_wd=half_wd)
    bincost.set_bits()

    for y in range(half_wd, half_wd+h):
        print (y)
        for x in range(half_wd, half_wd+w):
            top = y - half_wd
            left = x - half_wd
            bot = y + half_wd + 1
            right = x + half_wd + 1
            left_bits = bincost.left_bits[y-half_wd, x-half_wd]

            for shift in range(max_disp):
                if left - shift >= 0:
                    right_bits = bincost.right_bits[y-half_wd, x-half_wd-shift]
                    match_cost = np.sum(np.logical_xor(left_bits, right_bits))
                    cost_volume[y-half_wd, x-half_wd, shift] = match_cost
                    del right_bits, match_cost
                    #print (y-half_wd, x-half_wd, shift, match_cost)
            #print (np.argmin(cost_volume[y-half_wd, x-half_wd, :]))

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
        cost_volume[:, :, z] = ndimage.median_filter(cost_volume[:, :, z], 3)
    labels = np.argmin(cost_volume, axis=2) * scale_factor
    #tmp
    for i in range(max_disp):
        labels[:, i] = labels[:, max_disp+1]
    labels = ndimage.median_filter(labels, 3)
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

    #bsm = BSM(max_disp, scale_factor, 'tsukuba.png')
    #bsm.setPairDistr()
    #bsm.match(img_left, img_right)    

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
