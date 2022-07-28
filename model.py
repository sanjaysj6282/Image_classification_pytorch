import torch
import torch.nn as nn
import torch.nn.functional as F

#Class to define the model which we will use for training
#Stuff to fill in: The Architecture of your model, the forward function to define the forward pass
# NOTE!: You are NOT allowed to use pretrained models for this task

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        # Useful Link: https://pytorch.org/docs/stable/nn.html
        #------------ENTER YOUR MODEL HERE----------------#        

    def forward(self, x):
        #---------Assuming x to be the input to the model, define the forward pass-----------#
        return F.softmax(x)