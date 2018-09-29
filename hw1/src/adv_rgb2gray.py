import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import argparse
from os import listdir, makedirs
from os.path import join, exists
import time

parser = argparse.ArgumentParser()
parser.add_argument('--source_path', default='../testdata')
parser.add_argument('--target_path', default='../result')
parser.add_argument('--img_name', default='0a.png')
parser.add_argument('--check_ans', default=1, type=int)

def joint_bilateral_filter(jnt_img, src_img, sigma_color=0.05, sigma_space=1):

    start = time.time()
    # (390, 390, 3)
    jnt_img = jnt_img.astype(np.float32) / 255.
    src_img = src_img.astype(np.float32) / 255.

    h = src_img.shape[0]
    w = src_img.shape[1]
    radius = 3 * sigma_space

    tar_img = np.zeros((h, w, 3)).astype(np.float32)

    for y in range(0, h):
        for x in range(0, w):

            # define the searching range
            y_init = np.maximum(0, y-radius)
            y_end  = np.minimum(h, y+radius+1)
            x_init = np.maximum(0, x-radius)
            x_end  = np.minimum(w, x+radius+1)
            #print (y_init, y_end, x_init, x_end)

            src_patch_b = src_img[y_init:y_end, x_init:x_end, 0]
            src_patch_g = src_img[y_init:y_end, x_init:x_end, 1]
            src_patch_r = src_img[y_init:y_end, x_init:x_end, 2]

            jnt_patch_b = jnt_img[y_init:y_end, x_init:x_end, 0]
            jnt_patch_g = jnt_img[y_init:y_end, x_init:x_end, 1]
            jnt_patch_r = jnt_img[y_init:y_end, x_init:x_end, 2]

            patch_y, patch_x = np.meshgrid(np.arange(y_init-y, y_end-y), np.arange(x_init-x, x_end-x))
            space_patch = (patch_y * patch_y + patch_x * patch_x).T # for some reason...
            hs = np.exp(-space_patch / (2 * (sigma_space ** 2)))

            jnt_diff_b = np.multiply(jnt_patch_b - src_img[y][x][0], jnt_patch_b - src_img[y][x][0])
            jnt_diff_g = np.multiply(jnt_patch_g - src_img[y][x][1], jnt_patch_g - src_img[y][x][1])
            jnt_diff_r = np.multiply(jnt_patch_r - src_img[y][x][2], jnt_patch_r - src_img[y][x][2])
            jnt_diff   = jnt_diff_b + jnt_patch_g + jnt_patch_r
            hr = np.exp(-jnt_diff / (2 * (sigma_color ** 2)))

            #print (hr)
            #print (x, y, jnt_patch_b.shape, hs.shape, hr.shape, src_patch_b.shape)

            response_patch_b = np.multiply(np.multiply(hs, hr), src_patch_b) / (np.sum(np.multiply(hs, hr)) + 1e-8)
            response_patch_g = np.multiply(np.multiply(hs, hr), src_patch_g) / (np.sum(np.multiply(hs, hr)) + 1e-8)
            response_patch_r = np.multiply(np.multiply(hs, hr), src_patch_r) / (np.sum(np.multiply(hs, hr)) + 1e-8)

            tar_img[y_init:y_end, x_init:x_end, 0] += response_patch_b
            tar_img[y_init:y_end, x_init:x_end, 1] += response_patch_g
            tar_img[y_init:y_end, x_init:x_end, 2] += response_patch_r

    tar_img *= 255
    interval = time.time() - start
    print ("time: ", interval)
    return tar_img

def get_candidates():
    candidates = list()
    for i in range(0, 10):
        for j in range(0, 10-i):
            choice = (i, j, 10-i-j)
            candidates.append(choice)
    candidates = np.array(candidates).astype(np.float32) / 10.
    return candidates

# for BGR order 
def weight_rgb2gray(src_img, weights):
    src_img = src_img.astype(np.float32)
    img_b = src_img[:, :, 0]
    img_g = src_img[:, :, 1]
    img_r = src_img[:, :, 2]
    tar_img = weights[0] * img_r + weights[1] * img_g + weights[2] * img_b
    tar_img = tar_img.reshape(tar_img.shape[0], tar_img.shape[1], 1).astype(np.uint8)
    tar_img = np.repeat(tar_img, 3, axis=2)
    return tar_img

def local_score(diff):
    score = np.zeros(len(diff) - 2)
    for i in range(1, len(diff) - 1):
        if diff[i] < diff[i-1] and diff[i] < diff[i+1]:
            score[i-1] = 1
    return score

if __name__ == '__main__':
    args = parser.parse_args()

    if not exists (args.target_path):
        makedirs(args.target_path)
    if not exists (args.source_path):
        print ("Error: The image path doesn't exist")
    if not args.img_name.endswith('.png'):
        print ("Warning: the image is not a .png file")

    src_img_path = join(args.source_path, args.img_name)
    src_img = cv2.imread(src_img_path) # BGR

    img_name_y0 = args.img_name[:-4] + '_y0' + '.png'
    img_name_y1 = args.img_name[:-4] + '_y0' + '.png'
    img_name_y2 = args.img_name[:-4] + '_y0' + '.png'
    first_img_path = join(args.target_path, img_name_y0)
    second_img_path = join(args.target_path, img_name_y1)
    third_img_path = join(args.target_path, img_name_y2)

    candidates = get_candidates()
    diff = np.zeros((68, 9)).astype(np.float32)
    np.save(join(args.target_path, 'diff.npy'), diff)
    para_index = 0

    sigma_color = [0.05, 0.1, 0.2]
    sigma_space = [1, 2, 3]

    for sc in sigma_color:
        for ss in sigma_space:
            cand_index = 0
            for weight in candidates:
                print (weight)
                can_img = weight_rgb2gray(src_img, weight)
                res_img = joint_bilateral_filter(jnt_img=can_img, src_img=src_img, sigma_color=sc, sigma_space=ss)
                diff[cand_index, para_index] = (np.sum(np.abs(src_img - res_img)))
                cand_index += 1
                figname = join(args.target_path, args.img_name[:-4] + '_w0_' + str(weight[0]) + '_w1_' + str(weight[1]) + '_w2_' + str(weight[2]) + '.png')
                cv2.imwrite(figname, res_img)

                #res_img = cv2.ximgproc.jointBilateralFilter(src=src_img, joint=can_img, d=7, sigmaColor=0.1, sigmaSpace=3)
                #cv2.imwrite(first_img_path, res_img)
            para_index += 1

    np.save(join(args.target_path, 'diff.npy'), diff)
    diff[0, :]  = 100
    diff[-1, :] = 100 # consider boundary condition

    # plot and score
    para_index = 0
    score = np.zeros(66, 9)
    for sc in sigma_color:
        for ss in sigma_space:
            y = diff[1:-1, para_index]
            x = np.arange(1, 67)
            plt.plot(y, x)
            figname = join(args.target_path, 'diff_sc_' + str(sc) + '_ss_' + str(ss) + '.png')
            plt.savefig(figname)
            plt.close()

            score[:, para_index] = local_score(diff[para_index])
            para_index += 1

    score = np.sum(score, axis=1)
    plt.plot(score, x)
    plt.savefig(join(args.target_path, 'score.png'))
    plt.close()
