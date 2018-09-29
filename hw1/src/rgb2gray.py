import numpy as np
import cv2
import argparse
from os import listdir, makedirs
from os.path import join, exists

parser = argparse.ArgumentParser()
parser.add_argument('--source_path', default='../testdata')
parser.add_argument('--target_path', default='../result')
parser.add_argument('--img_name', default='0a.png')
parser.add_argument('--check_ans', default=1, type=int)

def my_rgb2gray(src_img, b=0, g=1, r=2):
    src_img = src_img.astype(np.float32)
    img_b = src_img[:, :, b]
    img_g = src_img[:, :, g]
    img_r = src_img[:, :, r]
    tar_img = 0.299 * img_r + 0.587 * img_g + 0.114 * img_b
    tar_img = tar_img.astype(np.uint8)
    return tar_img

if __name__ == '__main__':
    args = parser.parse_args()

    if not exists (args.target_path):
        makedirs(args.target_path)
    if not exists (args.source_path):
        print ("Error: The image path doesn't exist")
    if not args.img_name.endswith('.png'):
        print ("Warning: the image is not a .png file")

    src_img_path = join(args.source_path, args.img_name)
    img_name_y = args.img_name[:-4] + '_y' + '.png'
    tar_img_path = join(args.target_path, img_name_y)

    src_img = cv2.imread(src_img_path) # BGR
    tar_img = my_rgb2gray(src_img)
    cv2.imwrite(tar_img_path, tar_img)

    if args.check_ans == 1:
        check_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        img_name_z = args.img_name[:-4] + '_z' + '.png'
        check_img_path = join(args.target_path, img_name_z)
        cv2.imwrite(check_img_path, check_img)

