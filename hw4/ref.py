import numpy as np
import cv2
import sys
import time
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(img_left, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(img_right, cmap='gray')
# plt.show()

class BSM():
    def __init__(self, max_disp, scale_factor, out_path):
        self.patch_size = 26
        self.des_len = 4096
        self.std = 4
        self.max_disp = max_disp
        self.scale_factor = scale_factor
        self.out_path = out_path

    def setPairDistr(self):
        # 4096 x 2
        pair_seq = (np.random.normal(0., 4.0, self.des_len*4))
        num_pair = self.des_len
        p_samples = np.zeros((self.des_len, 2), dtype = np.float_)
        p_samples[:, 0] = pair_seq[:num_pair]
        p_samples[:, 1] = pair_seq[num_pair:num_pair*2]
        self.p_samples = p_samples

        q_samples = np.zeros((self.des_len, 2), dtype = np.float_)
        q_samples[:, 0] = pair_seq[num_pair*2:num_pair*3]
        q_samples[:, 1] = pair_seq[num_pair*3:]
        self.q_samples = q_samples

        self.p_samples = np.clip(self.p_samples.astype(np.int_), -13, 13)
        self.q_samples = np.clip(self.q_samples.astype(np.int_), -13, 13)
        print('finish setting pair distribution')

    def match(self, i1, i2):
        self.img_left = i1
        self.img_right = i2
        self.img_left = cv2.cvtColor(self.img_left, cv2.COLOR_BGR2GRAY)
        self.img_right = cv2.cvtColor(self.img_right, cv2.COLOR_BGR2GRAY)

        row, col = self.img_left.shape
        disparity_map = np.zeros((row, col), dtype = np.int_)
        for r in range(row):
            for c in range(col):
                source_bits = self._findbinstring(r, c, self.img_left)
                ham_dis = 1e9 # initialize
                disparity = None
                for d in range(self.max_disp):
                    if c - d >= 0: 
                        target_bits = self._findbinstring(r, c - d, self.img_right)
                        count = np.count_nonzero(source_bits ^ target_bits)
                        print (r, c, d, count)
                        time.sleep(0.1)
                        if count < ham_dis:
                            disparity = d
                            ham_dis = count
                print (ham_dis, disparity)
                disparity_map[r, c] = disparity * self.scale_factor
                cv2.imwrite(self.out_path, disparity_map)
            print(r)
        print(disparity_map)

    def _findbinstring(self, r_offset, c_offset, img):
        pad = self.patch_size // 2
        image_pad = np.pad(img, (self.patch_size // 2, self.patch_size // 2), 'constant', constant_values=(0, 0))
        p_r = self.p_samples[:, 0] + r_offset + pad
        p_c = self.p_samples[:, 1] + c_offset + pad
        q_r = self.q_samples[:, 0] + r_offset + pad
        q_c = self.q_samples[:, 1] + c_offset + pad
        bits = image_pad[p_r, p_c] > image_pad[q_r, q_c]

        return bits

if __name__ == '__main__':
    a = BSM(MAX_DISP, SCALE_FACTOR)
    a.setPairDistr()
    a.match(img_left, img_right)