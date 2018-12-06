import argparse

import numpy as np
import cv2
from os.path import exists

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", default='./video/ar_marker.mp4')
parser.add_argument("--marker_path", default='./video/marker.png')
parser.add_argument("--sticker_path", default='./video/kp.jpg')
parser.add_argument("--ar_path", default='./video/target.mp4')

class AR_Video():
    def __init__(self, args):
        self.video = cv2.VideoCapture(args.video_path)
        self.set_output(args.ar_path)

        self.marker = cv2.imread(args.marker_path)
        self.sticker = cv2.imread(args.sticker_path)
        self.sticker = cv2.resize(self.sticker, self.marker.shape[:2])

        self.surf = cv2.xfeatures2d.SURF_create()
        self.bf = cv2.BFMatcher()

        self.kp_m, self.des_m = self.surf.detectAndCompute(self.marker, None)
        self.min_match_count = 15

    def set_output(self, ar_path):
        width = int(self.video.get(3))
        height = int(self.video.get(4))
        fps = self.video.get(5)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.ar_video = cv2.VideoWriter(ar_path, fourcc, fps, (width, height))

    def destroy(self):
        self.video.release()
        self.ar_video.release()
        cv2.destroyAllWindows()

    def execute(self):
        count = 0
        while self.video.isOpened():
            ret, frame = self.video.read()
            print (count)

            #print (self.marker.shape, frame.shape)

            if not ret:
                break
            kp_f, des_f = self.surf.detectAndCompute(frame, None)
            matches = self.bf.knnMatch(self.des_m, des_f, k=2)
            good = list()

            for (m, n) in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) > self.min_match_count:
                src_pts = np.float32([ self.kp_m[m_.queryIdx].pt for m_ in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp_f[m_.trainIdx].pt for m_ in good ]).reshape(-1,1,2)
                homography, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)                

            s_h, s_w, _s_c = self.sticker.shape
            offset = 20

            for y in range(s_h):
                for x in range(s_w):
                    pixel = self.sticker[y, x, :]
                    u_vec = np.array([x, y, 1])
                    v_vec =  np.dot(homography, u_vec.T)#.astype(np.int16)
                    t_x = int(v_vec[0] / v_vec[2])
                    t_y = int(v_vec[1] / v_vec[2])

                    t_x = np.clip(t_x, 3, frame.shape[1] - 5)
                    t_y = np.clip(t_y, 3, frame.shape[0] - 5)
                    frame[t_y-3 : t_y+4, t_x-3 : t_x+4, :] = pixel
            self.ar_video.write(frame)
            count += 1

if __name__ == '__main__':
    args = parser.parse_args()
    AR = AR_Video(args)
    AR.execute()
    AR.destroy()





