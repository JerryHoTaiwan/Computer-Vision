from dataset import MyMnist
from model import Classifier
import numpy as np
import sys
import time
import argparse
from os.path import join

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.optim import *

class Tester():

    def __init__(self, test_datapath='hw2-3_data/', csv_path='predict_mnist.csv', model_path='models/classifier.pkl', batchsize=128):
        self.datapath = datapath
        self.test_data = MyMnist(test_datapath, is_test=1)
        self.test_x = self.test_data.img_data
        self.test_y = torch.zeros(len(test_x))

        self.model = torch.load(model_path)
        self.model_path = model_path

        self.imgname = list()
        for img_path in listdir(test_datapath):
            self.imgname.append(img_path[:4])

    def execute(self):
        self.test()
        self.test_y = self.test_y.numpy()
        self.write_csv()

    def test():
        self.model.eval()

        for index in range(0, len(self.test_x), self.batchsize):
            if index+self.batchsize > len(self.valid_x):
                input_X = Variable(self.test_x[index:].cuda())
            else:
                input_X = Variable(self.test_x[index:index+self.batchsize].cuda())

            y_hat = self.model(input_X).data.cpu()
            if index+self.batchsize > len(self.valid_x):
                self.test_y[index:] = y_hat
            else:
                self.test_y[index:index+self.batchsize] = y_hat

    def write_csv(self):
        f = open(self.csv_path, 'w')
        f.write('id, label\n')
        for i, img_index in enumerate(self.imgname):
            f.write(img_index + ',' + str(self.test_y[i]) + '\n')
        f.close()

if __name__ == '__main__':
    datapath = sys.argv[1]
    csv_path = sys.argv[2]
    trainer = Tester(datapath, csv_path)