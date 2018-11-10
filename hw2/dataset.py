import numpy as np
import torch
from os import listdir, makedirs
from os.path import join, exists
import cv2

class MyMnist:

    def __init__(self, datapath='./hw2-3_data/train/', is_test=0):
        self.datapath = datapath
        self.img_data = list()
        self.label = list()
        self.is_test = is_test
        self.get_data()
        self.normalize()
        self.img_data = torch.from_numpy(self.img_data)
        self.label = torch.from_numpy(self.label)
        #self.label = torch.zeros(len(self.label), 10).scatter_(1, self.label, 1).type(torch.LongTensor)

    def get_data(self):

        if self.is_test == 1:
            for img_path in listdir(self.datapath):
                img_path_full = join(self.datapath, img_path)
                img =  cv2.imread(img_path_full, cv2.IMREAD_GRAYSCALE)
                img = img.reshape(1, img.shape[0], img.shape[1])
                self.img_data.append(img)
                self.label.append(0)
        else:
            for class_folder in listdir(self.datapath):
                class_label = int(class_folder[-1])
                class_folder_path = join(self.datapath, class_folder)
                for img_path in listdir(class_folder_path):
                    img_path_full = join(class_folder_path, img_path)
                    img =  cv2.imread(img_path_full, cv2.IMREAD_GRAYSCALE)
                    img = img.reshape(1, img.shape[0], img.shape[1])
                    self.img_data.append(img)
                    self.label.append(class_label)

        self.img_data = np.array(self.img_data).astype(np.float64)
        self.label = np.array(self.label).astype(np.long)

    def normalize(self):
        img_avg = np.mean(self.img_data, axis=0)
        img_std = np.std(self.img_data, axis=0) + 10e-8
        self.img_data -= img_avg
        self.img_data /= img_std
