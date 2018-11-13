import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Function, Variable

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(5,5),stride=(1,1),padding=(0,0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(5,5),stride=(1,1),padding=(0,0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            )

        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,10)
            #nn.Softmax(1)
            )

    def forward(self,x):
        x = x.type(torch.cuda.FloatTensor)

        x = self.extractor(x)
        x = x.view(x.size(0),-1)
        output = self.fc(x)
    
        return output

class Classifier_Vis(nn.Module):
    def __init__(self, is_train=1):
        super(Classifier_Vis, self).__init__()
        self.is_train = is_train

        self.conv0_0 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(5,5),stride=(1,1),padding=(0,0)),
            )
        self.conv0_1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            )

        self.conv1_0 = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(5,5),stride=(1,1),padding=(0,0)),
            )

        self.conv1_1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            )


        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,10)
            #nn.Softmax(1)
            )

    def forward(self,x):
        x = x.type(torch.cuda.FloatTensor)

        x0_0 = self.conv0_0(x)
        x0_1 = self.conv0_1(x0_0)

        x1_0 = self.conv1_0(x0_1)
        x1_1 = self.conv1_1(x1_0)

        x2 = x1_1.view(x1_1.size(0),-1)
        output = self.fc(x2)
    
        if (self.is_train == 1):
            return output        
        else:
            return x0_1, x1_1, output