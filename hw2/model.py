import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Function, Variable

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(5,5),stride=(1,1),padding=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),
            )

        self.fc = nn.Sequential(
            nn.Linear(2304,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,64),
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