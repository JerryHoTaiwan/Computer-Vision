import numpy as np
import argparse
import cv2
import time
import os.path
from util import writePFM
from disp_mgr import dispMgr

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='../../data/Synthetic/TL3.png', type=str, help='input left image')
parser.add_argument('--input-right', default='../../data/Synthetic/TR3.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL3.pfm', type=str, help='left disparity map')

def main():
    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    DM = dispMgr(img_left,img_right)
    disp, outlier = DM.computeDisp()
    #cv2.imwrite('outlier/' + os.path.split(args.output)[1][:-3] + 'png', outlier)
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
