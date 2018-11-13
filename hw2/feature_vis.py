from dataset import MyMnist
from model import Classifier, Classifier_Vis

import sys
import time
import argparse
from os.path import join

import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.optim import *

def plot_embedding(data, label, fig_name):
    x = data[:, 0]
    y = data[:, 1]
    c = label
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(x, y, c=c, cmap=cm.tab10, linewidth=0, antialiased=False)
    fig.colorbar(scat, shrink=0.5, aspect=5)
    ax.set_title('TSNE')
    plt.savefig(fig_name)

class Plotter():

    def __init__(self, test_datapath='hw2-3_data/', model_path='models/classifier_vis.pkl', batchsize=128):
        self.datapath = datapath
        self.valid_data = MyMnist(join(self.datapath, 'valid'))

        self.valid_x = self.valid_data.img_data
        self.valid_y = self.valid_data.label

        self.low_feat = np.zeros((len(self.valid_x), 1152))
        self.high_feat = np.zeros((len(self.valid_x), 256))
        self.test_feat = np.zeros((len(self.valid_x), 1))

        self.model_path = model_path
        self.model = torch.load(model_path)
        self.model.is_train = 0
        self.batchsize = batchsize

    def execute(self):
        self.test()
        self.valid_y = self.valid_y.numpy()

        perm_index = torch.randperm(10000).numpy()[:1000]

        self.label_sfl = self.valid_y[perm_index]
        self.low_sfl = self.low_feat[perm_index]
        self.high_sfl = self.high_feat[perm_index]
        self.test_sfl = self.test_feat[perm_index] 

        low_embedded = TSNE(n_components=2).fit_transform(self.low_sfl)
        high_embedded = TSNE(n_components=2).fit_transform(self.high_sfl)

        plot_embedding(low_embedded , self.label_sfl, 'low_feature')
        plot_embedding(high_embedded, self.label_sfl, 'high_feature')

    def test(self):
        self.model.eval()

        for index in range(0, len(self.valid_x), self.batchsize):
            
            if index+self.batchsize > len(self.valid_x):
                input_X = Variable(self.valid_x[index:].cuda())
                input_Y = Variable(self.valid_y[index:].cuda())
            else:
                input_X = Variable(self.valid_x[index:index+self.batchsize].cuda())
                input_Y = Variable(self.valid_y[index:index+self.batchsize].cuda())

            low_f, high_f, _y_hat = self.model(input_X)

            low_f = low_f.view(low_f.size(0), -1).cpu().data.numpy()
            high_f = high_f.view(high_f.size(0), -1).cpu().data.numpy()
            _y_hat = torch.argmax(_y_hat, 1).cpu().data.numpy().reshape(-1, 1)
            
            if index+self.batchsize > len(self.valid_x):
                self.low_feat[index:] = low_f
                self.high_feat[index:] = high_f
                self.test_feat[index:] = _y_hat

            else:
                self.high_feat[index:index+self.batchsize] = high_f
                self.low_feat[index:index+self.batchsize] = low_f
                self.test_feat[index:index+self.batchsize] = _y_hat

if __name__ == '__main__':
    datapath = sys.argv[1]
    plotter = Plotter(datapath)
    plotter.execute()