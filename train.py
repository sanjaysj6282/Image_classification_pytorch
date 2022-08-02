from cProfile import run
from msilib.schema import Class
from dataloader import inaturalist
from model import Classifier
import torch.nn as nn
import torch.optim as optim
import os
import time
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

#  to import a pretrained model for testing
import torchvision.models

# Sections to Fill: Define Loss function, optimizer and model, Train and Eval functions and the training loop

############################################# DEFINE HYPERPARAMS #####################################################
# Feel free to change these hyperparams based on your machine's capactiy
batch_size = 32
epochs = 10
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

############################################# DEFINE DATALOADER #####################################################
trainset = inaturalist(root_dir='../nature_12K/inaturalist_12K', mode='train')
valset = inaturalist(root_dir='../nature_12K/inaturalist_12K', mode = 'val')

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

################################### DEFINE LOSS FUNCTION, MODEL AND OPTIMIZER ######################################
# USEFUL LINK: https://pytorch.org/docs/stable/nn.html#loss-functions
#---Define the loss function to use, model object and the optimizer for training---#

# different type of images
no_of_classes=10
# cross entropy loss is better for classification problems
loss=nn.CrossEntropyLoss()
#  to --> if cuda(here present) then use cuda or else cpu
# model=Classifier(no_of_classes).to(device)
model=torchvision.models.resnet18()
no_features=model.fc.in_features
model.fc=nn.Linear(no_features, no_of_classes)
model=model.to(device)

# usual momentum=0.9 to converge faster
optimizer=optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

################################### CREATE CHECKPOINT DIRECTORY ####################################################
# NOTE: If you are using Kaggle to train this, remove this section. Kaggle doesn't allow creating new directories.
checkpoint_dir = 'checkpoints'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

#################################### HELPER FUNCTIONS ##############################################################

# def get_model_summary(model, input_tensor_shape):
#     summary(model, input_tensor_shape)

def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred.data, 1)
    total = y.size(0) # total no of images
    correct = (predicted == y).sum().item()
    return [correct, total]

def train(model, dataset, optimizer, criterion, device):
    '''
    Write the function to train the model for one epoch
    Feel free to use the accuracy function defined above as an extra metric to track
    '''
    model.train()
    correct=0
    total=0
    #------YOUR CODE HERE-----#
    for img_data in dataset:
        img, label=img_data
        img=img.to(device)
        label=label-1
        label=label.to(device)
        
        #  IMP
        # set gradients to zero
        optimizer.zero_grad()
        
        # current output after training
        curr_output=model(img)
        
        total+=label.size(0)
        _, predicted = torch.max(curr_output.data, 1)
            
        # print(predicted, label)
        correct+=(predicted == label).sum().item()
        # print(correct, total)
        
        # modify loss
        curr_loss=criterion(curr_output, label)
        # back propogation
        curr_loss.backward()
        
    ans=100.00*correct/total
    print("Current accuracy Traning :" +str(ans))
      
    

def eval(model, dataset, criterion, device):
    '''
    Write the function to validate the model after each epoch
    Feel free to use the accuracy function defined above as an extra metric to track
    '''
    #------YOUR CODE HERE-----#
    model.eval()
    run_correct=0
    curr_total=0
    with torch.no_grad():
        for img_data in dataset:
            img, label=img_data
            img=img.to(device)
            label=label.to(device)
                
            # set gradients to zero
            optimizer.zero_grad()
            
            # current output after evaluating
            curr_output=model(img)
            # print("Label:"+str(curr_output))
            
            curr_total+=label.size(0)
            _, predicted = torch.max(curr_output.data, 1)
            run_correct+=(predicted == label).sum().item()
        
    ans=100.00*run_correct/curr_total
    print("Current accuracy in Evaluation:" +str(ans))        

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

################################################### TRAINING #######################################################
# Get model Summary
# get_model_summary(model, [3, 256, 256])

#Training and Validation
best_valid_loss = float('inf')

def main():
    for epoch in range(epochs):
        start_time = time.monotonic()
        
        print("Epoch "+str(epoch+1))
        
        '''
        Insert code to train and evaluate the model (Hint: use the functions you previously made :P)
        Also save the weights of the model in the checkpoint directory
        '''
        #------YOUR CODE HERE-----#
        train(model, trainloader, optimizer, loss, device)
        
        eval(model, valloader, loss, device)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print("\nTIME TAKEN FOR THE EPOCH {}: {} mins and {} seconds".format(epoch+1, epoch_mins, epoch_secs))


    print("OVERALL TRAINING COMPLETE")
    
if __name__ == '__main__':
    main()