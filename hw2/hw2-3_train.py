from dataset import MyMnist
from model import Classifier, Classifier_Vis
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

class Trainer():

    def __init__(self, datapath='hw2-3_data/', model_path='models/classifier.pkl', patience=10, batchsize=128, max_epoch=100):
        self.datapath = datapath
        self.train_data = MyMnist(join(self.datapath, 'train'))
        self.valid_data = MyMnist(join(self.datapath, 'valid'))

        self.train_x = self.train_data.img_data
        self.train_y = self.train_data.label
        self.valid_x = self.valid_data.img_data
        self.valid_y = self.valid_data.label

        #self.model = Classifier_Vis().cuda()
        self.model = Classifier().cuda()

        self.model_path = model_path
        self.batchsize = batchsize
        self.max_epoch = max_epoch
        self.patience = patience

        self.train_loss_curv = list()
        self.train_acc_curv = list()
        self.valid_loss_curv = list()
        self.valid_acc_curv = list()

    def setting(self):

        self.optimizer = torch.optim.Adamax(self.model.parameters())
        self.loss_func = nn.CrossEntropyLoss()
        
        self.Max_loss = 1000000.0
        self.total_length = len(self.train_x)

        ### training ###
        self.count = 0

    def execute(self):
        self.setting()
        for epoch in range(self.max_epoch):
            self.train(epoch=epoch)
            self.eval()
            if (self.count == self.patience):
                break

        self.epoch = epoch

        self.plot_curve(np.array(self.train_loss_curv), 'training_loss')
        self.plot_curve(np.array(self.train_acc_curv), 'training_acc')
        self.plot_curve(np.array(self.valid_loss_curv), 'valid_loss')
        self.plot_curve(np.array(self.valid_acc_curv), 'valid_acc')

    def train(self, epoch):
        print("Epoch:", epoch+1)

        running_loss = 0.0
        running_acc = 0.0

        # shuffle
        perm_index = torch.randperm(self.total_length)
        train_X_sfl = self.train_x[perm_index]
        train_Y_sfl = self.train_y[perm_index]
            
        # construct training batch
        for index in range(0, self.total_length, self.batchsize):
            if index+self.batchsize > self.total_length:
                break

            # zero the parameter gradients
            input_X = Variable(train_X_sfl[index:index+self.batchsize]).cuda()
            input_y = Variable(train_Y_sfl[index:index+self.batchsize]).cuda()

            # forward + backward + optimize
            y_hat = self.model(input_X)

            self.optimizer.zero_grad()
            loss = self.loss_func(y_hat, input_y)
            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            
            output_label = torch.argmax(y_hat,1).cpu().data
            ans_y = input_y.cpu().data
            running_acc += np.sum((output_label == ans_y).numpy()) 

        print("Training Loss: ", running_loss/(self.total_length/self.batchsize))
        print("Training Acc: ", running_acc/(self.total_length))

        self.train_loss_curv.append(running_loss/(self.total_length/self.batchsize))
        self.train_acc_curv.append(running_acc/(self.total_length))

    def eval(self):
        self.model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        valid_index = torch.randperm(len(self.valid_x))
        valid_X_sfl = self.valid_x[valid_index]
        valid_Y_sfl = self.valid_y[valid_index]

        for index in range(0,len(self.valid_x) ,self.batchsize):
            if index+self.batchsize > len(self.valid_x):
                input_X = Variable(valid_X_sfl[index:].cuda())
                input_Y = Variable(valid_Y_sfl[index:].cuda())
            else:
                input_X = Variable(valid_X_sfl[index:index+self.batchsize].cuda())
                input_Y = Variable(valid_Y_sfl[index:index+self.batchsize].cuda())

            y_hat = self.model(input_X)
            loss = self.loss_func(y_hat, input_Y)
            valid_loss += loss.item()#.data[0]

            output_label = torch.argmax(y_hat,1).cpu().data
            ans_y = input_Y.cpu().data
            valid_acc += np.sum((output_label == ans_y).numpy())

        if (valid_loss/(len(self.valid_x)/self.batchsize) < self.Max_loss):
            self.Max_loss = valid_loss/(len(self.valid_x)/self.batchsize)
            print ("The loss is improved to : ",valid_loss/(len(self.valid_x)/self.batchsize)," \nsaving model...")
            torch.save(self.model, self.model_path)
            self.count = 0
        else:
            print("Validation Loss: ",valid_loss/(len(self.valid_x)/self.batchsize), "doesn't improve")
            self.count += 1
        
        print ("Validation Acc: ", valid_acc/(len(self.valid_x))) 
        
        self.valid_loss_curv.append(valid_loss/(len(self.valid_x)/self.batchsize))
        self.valid_acc_curv.append(valid_acc/(len(self.valid_x)))
        
        self.model.train()

    def plot_curve(self, data, y_label):
        x_epoch = np.arange(self.epoch+1)
        plt.plot(x_epoch, data)
        plt.xlabel('epoch')
        plt.ylabel(y_label)
        plt.savefig(y_label+'.png')
        plt.close()

if __name__ == '__main__':
    datapath = sys.argv[1]
    trainer = Trainer(datapath)
    trainer.execute()