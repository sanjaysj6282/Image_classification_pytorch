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
epochs = 20
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

############################################# DEFINE DATALOADER #####################################################
trainset = inaturalist(root_dir='../nature_12K/inaturalist_12K', mode='train')
valset = inaturalist(root_dir='../nature_12K/inaturalist_12K', mode = 'val')
# trainset = inaturalist(root_dir='../monkey_dataset', mode='train')
# valset = inaturalist(root_dir='../monkey_dataset', mode = 'val')

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

################################### DEFINE LOSS FUNCTION, MODEL AND OPTIMIZER ######################################
# USEFUL LINK: https://pytorch.org/docs/stable/nn.html#loss-functions
#---Define the loss function to use, model object and the optimizer for training---#

# different type of images
no_of_classes=10
# cross entropy loss is better for classification problems
loss_fn=nn.CrossEntropyLoss()
#  to --> if cuda(here present) then use cuda or else cpu
# model=Classifier(no_of_classes).to(device)
model=torchvision.models.resnet50(weights=None) # Resnet is trained on imagenet
no_features=model.fc.in_features
model.fc=nn.Linear(no_features, no_of_classes)
model=model.to(device)

# usual momentum=0.9 to converge faster
# optimizer=optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.03)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Path for checkpoint
path='./checkpoints/checkpoints.pt'

################################### CREATE CHECKPOINT DIRECTORY ####################################################
# NOTE: If you are using Kaggle to train this, remove this section. Kaggle doesn't allow creating new directories.
checkpoint_dir = 'checkpoints'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

#################################### HELPER FUNCTIONS ##############################################################

def get_model_summary(model, input_tensor_shape):
    summary(model, input_tensor_shape)
    print("\n\n")

def train(model, dataset, optimizer, criterion, device):
    model.train()
    correct=0
    total=0
    for img_data in dataset:
        img, label=img_data
        img=img.to(device)
        label=label-1
        label=label.to(device)
        
        #  IMP
        # set gradients to zero
        optimizer.zero_grad()
        
        # current output after training
        output=model(img)
        
        total+=label.size(0)
        _, predicted = torch.max(output.data, 1)
            
        correct+=(predicted == label).sum().item()
        # print(predicted, label)
        # print(correct, total)
        
        # modify loss
        loss=criterion(output, label)
        # back propogation
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        
    accuracy=100.00*correct/total
    print("Accuracy in Traning :" +str(accuracy))
    
      
    
def eval(model, dataset, device, best_accuracy, epoch_now):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for img_data in dataset:
            img, label=img_data
            img=img.to(device)
            label=label-1
            label=label.to(device)
                
            # set gradients to zero
            optimizer.zero_grad()
            
            # current output after evaluating
            output=model(img)
            
            total+=label.size(0)
            _, predicted = torch.max(output.data, 1)
            correct+=(predicted == label).sum().item()
            # print(predicted, label)
            # print(correct, total)
        
    accuracy=100.00*correct/total
    print("Accuracy in Evaluation:" +str(accuracy))     
    
    if accuracy > best_accuracy:
        best_accuracy=accuracy
        torch.save({
            'epoch': epoch_now,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path)
        print("Checkpoint saved")
    print("Best Accuracy in Evaluation:" +str(accuracy)+"\n")      
    
    return best_accuracy

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

################################################### TRAINING #######################################################

#Training and Validation
best_valid_loss = float('inf')

def main():
    # Get model Summary
    get_model_summary(model, (3, 256, 256))
    
    # curr_epoch=0
    best_accuracy=0
    for epoch in range(epochs):
        start_time = time.monotonic()
        
        #------YOUR CODE HERE-----#
        print("Epoch "+str(epoch+1))
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            curr_epoch = checkpoint['epoch']
            # print("Epoch saved upto "+str(curr_epoch))
            print("Checkpoint is loaded\n")
        train(model, trainloader, optimizer, loss_fn, device)
        best_accuracy=eval(model, valloader, device, best_accuracy, epoch+1)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print("TIME TAKEN FOR THE EPOCH {}: {} mins and {} seconds\n\n".format(epoch+1, epoch_mins, epoch_secs))

    print("OVERALL TRAINING COMPLETE")
    
if __name__ == '__main__':
    main()