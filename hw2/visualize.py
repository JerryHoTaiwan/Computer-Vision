"""
Created on Sat Nov 18 23:12:08 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models
from torch.autograd import Variable

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='./models/classifier.pkl')
parser.add_argument("--cnn_layer", default=5, type=int)
parser.add_argument("--filter_pos", default=20, type=int)

def eigen2img(img):
        img = img * 1.0
        img -= np.min(img)
        img /= np.max(img)
        img = (img * 255).astype(np.uint8)
        return img

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = np.float64(np.random.uniform(0, 1, (1, 1, 28, 28)))
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Process image and return variable
        #self.processed_image, self.im_min, self.im_max = preprocess_image(self.created_image)
        # Define optimizer for the image
        self.processed_image = torch.from_numpy(self.created_image.astype(np.float64))
        self.processed_image = Variable(self.processed_image, requires_grad=True).cpu()

        optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        print ("start!")
        for i in range(1, 101):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                print (index)
                x = x.type(torch.cuda.FloatTensor)
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            
            # Update image
            optimizer.step()

            # Recreate image

            self.created_image = self.processed_image.data.cpu().numpy().reshape(28, 28)
            self.created_image = eigen2img(self.created_image)
            # Save image
            if i % 20 == 0:
                cv2.imwrite('../generated/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg',
                            self.created_image)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        self.processed_image = self.created_image
        
        #self.processed_image = preprocess_image(self.created_image)
        # Define optimizer for the image
        optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = self.processed_image
            # Save image
            if i % 5 == 0:
                cv2.imwrite('../generated/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg',
                            self.created_image)


if __name__ == '__main__':
    args = parser.parse_args()

    cnn_layer = args.cnn_layer
    filter_pos = args.filter_pos
    # Fully connected layer is not needed
    #pretrained_model = models.vgg16(pretrained=True).features
    pretrained_model = torch.load(args.model_path).extractor
    #for index, layer in enumerate(pretrained_model):
        #print (index, layer)

    for filt_index in range(filter_pos):
        layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filt_index)

        # Layer visualization with pytorch hooks
        layer_vis.visualise_layer_with_hooks()

        # Layer visualization without pytorch hooks
        # layer_vis.visualise_layer_without_hooks()