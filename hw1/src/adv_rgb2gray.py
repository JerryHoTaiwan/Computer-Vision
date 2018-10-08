import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import cv2
import argparse
from os import listdir, makedirs
from os.path import join, exists
import time

#from bennyjbf import JointBilateralFilter

parser = argparse.ArgumentParser()
parser.add_argument('--source_path', default='../testdata')
parser.add_argument('--target_path', default='../result')
parser.add_argument('--img_name', default='0a.png')
parser.add_argument('--check_ans', default=1, type=int)
parser.add_argument('--already_done', default=0, type=int)
parser.add_argument('--r_factor', default=3, type=float)
parser.add_argument('--use_ans', default=0, type=int)
parser.add_argument('--save_img', default=0, type=int)

def joint_bilateral_filter(jnt_img, src_img, sigma_color=0.05, sigma_space=1, r_factor=1.5):

    start = time.time()

    # (390, 390, 3)
    jnt_img = jnt_img.astype(np.float32)
    src_img = src_img.astype(np.float32)
    radius = np.round(r_factor * sigma_space).astype(np.int16)
    sigma_color *= 255

    jnt_pad = cv2.copyMakeBorder(jnt_img, radius, radius, radius, radius, cv2.BORDER_REFLECT).astype(np.float32)
    jnt_pad = jnt_pad.reshape(jnt_pad.shape[0], jnt_pad.shape[1], jnt_img.shape[2]) # avoid cancelling the dimension
    src_pad = cv2.copyMakeBorder(src_img, radius, radius, radius, radius, cv2.BORDER_REFLECT).astype(np.float32)

    h, w = src_img.shape[0], src_img.shape[1]
    tar_img = np.zeros((h+2*radius, w+2*radius, 3)).astype(np.float32)

    patch_y, patch_x = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    space_patch = patch_y * patch_y + patch_x * patch_x
    hs = np.exp(-space_patch / (2 * (sigma_space ** 2)))

    for y in range(radius, h + radius):
        for x in range(radius, w + radius):

            src_patch = src_pad[(y-radius):(y+radius+1), (x-radius):(x+radius+1), :]
            jnt_patch = jnt_pad[(y-radius):(y+radius+1), (x-radius):(x+radius+1), :]

            if (jnt_patch.shape[2] == 1):
                jnt_diff  = np.multiply(jnt_patch - jnt_pad[y][x][0], jnt_patch - jnt_pad[y][x][0])
                hr = np.exp(-jnt_diff / (2 * (sigma_color ** 2))).reshape((2*radius+1, 2*radius+1))

            elif (jnt_patch.shape[2] == 3):

                jnt_diff = np.multiply(jnt_patch - jnt_pad[y][x], jnt_patch - jnt_pad[y][x])
                jnt_exp_diff = np.exp(-jnt_diff / (2 * (sigma_color ** 2))).reshape((2*radius+1, 2*radius+1, 3))
                hr = jnt_exp_diff[:, :, 0] * jnt_exp_diff[:, :, 1] * jnt_exp_diff[:, :, 2]

            # weighted sum
            W = (np.sum(np.multiply(hs, hr))) + 1e-8
            response_b = np.sum(np.multiply(np.multiply(hs, hr), src_patch[:, :, 0])) / W
            response_g = np.sum(np.multiply(np.multiply(hs, hr), src_patch[:, :, 1])) / W
            response_r = np.sum(np.multiply(np.multiply(hs, hr), src_patch[:, :, 2])) / W

            tar_img[y, x, 0] = response_b
            tar_img[y, x, 1] = response_g
            tar_img[y, x, 2] = response_r

    tar_img = tar_img[radius:(h+radius), radius:(w+radius)] #* 255
    interval = time.time() - start
    print ("time: ", interval)
    return tar_img#, check

def get_candidates():
    candidates = list()
    for i in range(0, 11):
        for j in range(0, 11-i):
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
    tar_img = weights[0] * img_b + weights[1] * img_g + weights[2] * img_r
    tar_img = tar_img.reshape(tar_img.shape[0], tar_img.shape[1], 1).astype(np.int16)
    return tar_img

def local_score(cand, diff):
    score = np.zeros(len(diff))
    for i, choice in enumerate(cand):
        local_min = 1
        for j, neibor in enumerate(cand):
            # the minimum distance is sqrt(2)
            if np.sum(np.abs(neibor - choice)) < 0.21 and np.sum(np.abs(neibor - choice)) > 0:
                if diff[j] > diff[i]:
                    local_min *= 1
                else:
                    local_min *= 0
        score[i] += local_min
    return score

def plot_4dsurface(candidates, diff, tar_img_path):

    scat_path = join(tar_img_path, 'surface')
    if not exists(scat_path):
        makedirs(scat_path)
    for para_index in range(9):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        c = diff[:, para_index]
        x = candidates[:, 0]
        y = candidates[:, 1]
        z = candidates[:, 2]
        scat = ax.scatter(x, y, z, c=c, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('blue')
        ax.set_ylabel('green')
        ax.set_zlabel('red')

        # Add a color bar which maps values to colors.
        fig.colorbar(scat, shrink=0.5, aspect=5)
        ax.view_init(30, 0)
        plt.draw()
        plt.savefig(join(scat_path, 'sf_' + str(para_index) + '.png'))
        plt.close()

if __name__ == '__main__':
    args = parser.parse_args()

    tar_img_path = join(args.target_path, args.img_name[:-4])
    tar_img_res_path = join(tar_img_path, 'res')
    tar_img_can_path = join(tar_img_path, 'can')

    if not exists (args.target_path):
        makedirs(args.target_path)
    if not exists(tar_img_path):
        makedirs(tar_img_path)
    if not exists(tar_img_res_path):
        makedirs(tar_img_res_path)
    if not exists(tar_img_can_path):
        makedirs(tar_img_can_path)

    if not exists (args.source_path):
        print ("Error: The image path doesn't exist")
    if not args.img_name.endswith('.png'):
        print ("Warning: the image is not a .png file")

    src_img_path = join(args.source_path, args.img_name)
    src_img = cv2.imread(src_img_path) # BGR

    img_name_y0 = args.img_name[:-4] + '_y0' + '.png'
    img_name_y1 = args.img_name[:-4] + '_y1' + '.png'
    img_name_y2 = args.img_name[:-4] + '_y2' + '.png'
    first_img_path = join(args.target_path, img_name_y0)
    second_img_path = join(args.target_path, img_name_y1)
    third_img_path = join(args.target_path, img_name_y2)

    candidates = get_candidates()
    diff = np.zeros((66, 9)).astype(np.float32)
    para_index = 0

    sigma_color = [0.05, 0.1, 0.2]
    sigma_space = [1, 2, 3]

    if args.already_done == 0:
        for sc in sigma_color:
            for ss in sigma_space:
                if args.use_ans == 1:
                    ref_img = cv2.ximgproc.jointBilateralFilter(src=src_img, joint=src_img, d=2*int(args.r_factor*ss)+1, sigmaColor=sc*255, sigmaSpace=ss)
                else:
                    ans_img = cv2.ximgproc.jointBilateralFilter(src=src_img, joint=src_img, d=2*int(args.r_factor*ss)+1, sigmaColor=sc*255, sigmaSpace=ss)
                    ref_img = joint_bilateral_filter(jnt_img=src_img, src_img=src_img, sigma_color=sc, sigma_space=ss, r_factor=args.r_factor)
                    #chk_img = JointBilateralFilter(ss=ss, sr=sc, guid=src_img, target=src_img)
                    print ("difference: ", np.sum(np.abs(ref_img - ans_img)) / (ref_img.size))
                    #print ("difference: ", np.sum(np.abs(chk_img - ans_img)) / (ref_img.size))

                cand_index = 0
                for weight in candidates:
                    print (weight)
                    can_img = weight_rgb2gray(src_img, weight)
                    can_figname = join(tar_img_can_path, args.img_name[:-4] + '_w0_' + str(weight[0]) + '_w1_' + str(weight[1]) + '_w2_' + str(weight[2]) + '.png')
                    res_figname = join(tar_img_res_path, args.img_name[:-4] + '_w0_' + str(weight[0]) + '_w1_' + str(weight[1]) + '_w2_' + str(weight[2]) + '.png')

                    if args.use_ans == 1:
                        can_img = can_img.astype(np.uint8)
                        res_img = cv2.ximgproc.jointBilateralFilter(src=src_img, joint=can_img, d=2*int(args.r_factor*ss)+1, sigmaColor=sc*255, sigmaSpace=ss)
                        ans_img = cv2.ximgproc.jointBilateralFilter(src=src_img, joint=can_img, d=2*int(args.r_factor*ss)+1, sigmaColor=sc*255, sigmaSpace=ss)

                    else:
                        #chk_img = JointBilateralFilter(ss=ss, sr=sc, guid=can_img, target=src_img)
                        res_img = joint_bilateral_filter(jnt_img=can_img, src_img=src_img, sigma_color=sc, sigma_space=ss, r_factor=args.r_factor)
                        can_img = can_img.astype(np.uint8)
                        ans_img = cv2.ximgproc.jointBilateralFilter(src=src_img, joint=can_img, d=2*int(args.r_factor*ss)+1, sigmaColor=sc*255, sigmaSpace=ss)
                        print ("difference: ", np.sum(np.abs(res_img - ans_img)) / (ref_img.size))
                        #print ("difference: ", np.sum(np.abs(chk_img - ans_img)) / (ref_img.size))


                    diff[cand_index, para_index] = (np.sum(np.abs(ref_img - res_img)))
                    cand_index += 1
                    if args.save_img:
                        cv2.imwrite(can_figname, can_img)
                        cv2.imwrite(res_figname, res_img)
                para_index += 1

        np.save(join(tar_img_path, 'diff.npy'), diff)

    else:
        diff = np.load(join(tar_img_path, 'diff.npy'))

    plot_4dsurface(candidates, diff, tar_img_path)

    # plot and score
    para_index = 0
    score = np.zeros((66, 9))
    for sc in sigma_color:
        for ss in sigma_space:
            score[:, para_index] = local_score(candidates, diff[:, para_index])
            para_index += 1
    x = np.arange(0, 66)
    score = np.sum(score, axis=1)
    plt.plot(x, score)
    plt.savefig(join(tar_img_path, 'score.png'))
    plt.close()

    # get images
    top_score = np.sort(score)[-3:]
    for i, value in enumerate(top_score):
        for j, number in enumerate(score):
            if number == value:
                score[j] == 0
                weight = candidates[j]
                print (weight)
                can_img = weight_rgb2gray(src_img, weight)

                if args.use_ans == 1:
                    can_img = can_img.astype(np.uint8)
                    res_img = cv2.ximgproc.jointBilateralFilter(src=src_img, joint=can_img, d=2*int(args.r_factor*ss)+1, sigmaColor=sc*255, sigmaSpace=ss)
                else:
                    res_img = joint_bilateral_filter(jnt_img=can_img, src_img=src_img, sigma_color=sc, sigma_space=ss, r_factor=args.r_factor)

                can_figname = join(tar_img_can_path, args.img_name[:-4] + '_y' + str(i) + '_.png')
                res_figname = join(tar_img_res_path, args.img_name[:-4] + '_y' + str(i) + '_.png')
                cv2.imwrite(can_figname, can_img)
                cv2.imwrite(res_figname, res_img)                    

                break
