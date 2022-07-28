from dataloader import inaturalist
from model import Classifier
import torch.nn as nn
import torch.optim as optim
import os
import time
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

# Sections to Fill: Define Loss function, optimizer and model, Train and Eval functions and the training loop

############################################# DEFINE HYPERPARAMS #####################################################
# Feel free to change these hyperparams based on your machine's capactiy
batch_size = 32
epochs = 10
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################# DEFINE DATALOADER #####################################################
trainset = inaturalist(root_dir='Data/inaturalist_12K', mode='train')
valset = inaturalist(root_dir='Data/inaturalist_12K', mode = 'val')

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)

################################### DEFINE LOSS FUNCTION, MODEL AND OPTIMIZER ######################################
# USEFUL LINK: https://pytorch.org/docs/stable/nn.html#loss-functions
#---Define the loss function to use, model object and the optimizer for training---#


################################### CREATE CHECKPOINT DIRECTORY ####################################################

# NOTE: If you are using Kaggle to train this, remove this section. Kaggle doesn't allow creating new directories.
checkpoint_dir = 'checkpoints'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

#################################### HELPER FUNCTIONS ##############################################################

def get_model_summary(model, input_tensor_shape):
    summary(model, input_tensor_shape)

def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return correct/total

def train(model, dataset, optimizer, criterion, device):
    '''
    Write the function to train the model for one epoch
    Feel free to use the accuracy function defined above as an extra metric to track
    '''
    #------YOUR CODE HERE-----#

def eval(model, dataset, criterion, device):
    '''
    Write the function to validate the model after each epoch
    Feel free to use the accuracy function defined above as an extra metric to track
    '''
    #------YOUR CODE HERE-----#

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

################################################### TRAINING #######################################################
# Get model Summary
get_model_summary(model, [3, 256, 256])

#Training and Validation
best_valid_loss = float('inf')

for epoch in range(epochs):
    
    start_time = time.monotonic()
    
    '''
    Insert code to train and evaluate the model (Hint: use the functions you previously made :P)
    Also save the weights of the model in the checkpoint directory
    '''
    #------YOUR CODE HERE-----#

    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print("\n\n\n TIME TAKEN FOR THE EPOCH: {} mins and {} seconds".format(epoch_mins, epoch_secs))


print("OVERALL TRAINING COMPLETE")