# Imports here
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
import helperUdacity
# I did this because I had problems with loading, but in the end, changing the recurssion limit did not do the trick
import sys
# Set recurssopm limit
sys.setrecursionlimit(10**5)  
# Import the file with the DL functions
import dlfunctions
# Import the library to interact with the command window
import argparse
from workspace_utils import active_session

###############################################################################################################################################
############# SET UP VARIABLES FOR THE TERMINAL INPUT
###############################################################################################################################################

parser = argparse.ArgumentParser(description="Train a deep neural network.")

# Input variables
parser.add_argument("data_dir", type=str, help="The name of the local directory where the train, test, and validation directories are.")

# Flags/options
parser.add_argument("-s", "--save_model", action="store_true", help='It will save a checkpoint after the trining.')
parser.add_argument("-gpu", action="store_true",
                    help="Uses the gpu for training. By default it will use the cpu.")
# I give more options for functionality. However, I used for the training and it worked vgg16, so my testing of the application will be based on that. It would be very easy to extend to all the possibilities in torchvision, as I fetch the infeatures dynamically.
parser.add_argument('-arch', "--architecture", type=str, choices=['vgg13', 'vgg16'], help="The name of the architecture to be used for the convolutional section of the deep neural network. Youy only have a set of options.")
parser.add_argument('-neurons', '--nargs_neurons_layer_list', nargs='+', type=int, help=' Same size as droput list. Provide list containing the number of neurons fir each layer.')
parser.add_argument('-dropouts', '--nargs_dropout_layer_list', nargs='+', type=float, help=' Same size as neurons list. Provide list containing the dropout probabilities for each layer.')
parser.add_argument('-lr', "--learning_rate", type=float, help="The learning rate of the model. lr= 0.001 is recommended.")
parser.add_argument('-e', "--epochs", type=int, help="The number of epochs for training.")

args = parser.parse_args()

###############################################################################################################################################
############# MAIN
###############################################################################################################################################

# Get the labeled list and the nmber of flowers
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
labels = []
for key, item in enumerate(cat_to_name):
    labels.append(int(item))
output_size = max(labels)

# Before training, there is no need for using the GPU. SO until then, we will use the cpu
device = 'cpu'

# Get the images ready to be used
trainloader, testloader, validloader, train_data = dlfunctions.image_pre_processing(args.data_dir)

# These condition statements allow for the user to choose what he wants and if she forgets a flag, then there will be a default value
# Build the deep nn
if args.architecture != None:
    if args.architecture == 'vgg13':
        print("We will use vgg13 for transfer learning")
        DL_model_virgin = models.vgg13(pretrained=True)
    elif args.architecture == 'vgg16':
        print("We will use vgg16 for transfer learning")
        DL_model_virgin = models.vgg16(pretrained=True)
        hidden_layers = args.nargs_neurons_layer_list
        dropout_probabilities = args.nargs_dropout_layer_list
        epochs = args.epochs
        learning_rate = args.learning_rate
else:
    print("We will use vgg16 for transfer learning as default")
    DL_model_virgin = models.vgg16(pretrained=True)

if args.nargs_neurons_layer_list != None:    
    hidden_layers = args.nargs_neurons_layer_list
else: 
    hidden_layers = [450]

dropout_probabilities = []
if args.nargs_dropout_layer_list != None:    
    dropout_probabilities = args.nargs_dropout_layer_list
else: 
    for i in hidden_layers:
        dropout_probabilities.append(0.25)
    
if args.epochs != None:    
    epochs = args.epochs
else: 
    epochs = 2

if args.learning_rate != None:    
    learning_rate = args.learning_rate
else: 
    learning_rate = 0.001


Classifier_input_size = DL_model_virgin.classifier[0].in_features 
Classifier_output_size = output_size # 102 in the flower dataset dataset

                         
DL_model = dlfunctions.Network(DL_model_virgin, Classifier_input_size, Classifier_output_size, hidden_layers, dropout_probabilities, device)

# We set the criterion and the optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(DL_model.classifier.parameters(), lr=learning_rate)
 
print(DL_model)
print('Directory of training data:', args.data_dir)
print('epochs', epochs)
print('lr', learning_rate)

if args.gpu:
    print("GPU mode activated for training")
    device = 'cuda'
else:
    print("CPU mode activated for training")
    device = 'cpu'

print_every = 30
dlfunctions.train_model(DL_model, trainloader, validloader, print_every, epochs, criterion, optimizer, device)
dlfunctions.validate_model(DL_model, testloader, criterion, device)

checkpoint = dlfunctions.create_checkpoint(DL_model, optimizer, train_data, Classifier_input_size, Classifier_output_size, \
                               hidden_layers, epochs, dropout_probabilities, criterion)

if args.save_model:
    print("Saving model as checkpoint2.pth")
    torch.save(checkpoint, 'checkpoint2.pth')

#You may try:
# python3 train.py flowers -arch  vgg16 -neurons 450 200 150  -dropouts 0.25 0.3 0.4  -lr 0.001 -e 10
# python3 train.py flowers -arch  vgg16 -neurons 450 200 150  
# python3 train.py flowers -s -gpu
