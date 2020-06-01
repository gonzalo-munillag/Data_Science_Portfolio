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

parser = argparse.ArgumentParser(description="Predict the class of an image.")


# Input variables
parser.add_argument("image_data_dir", type=str, help="The name of the local directory where the image is.")
parser.add_argument("checkpoint_dir", type=str, help="The name of the local directory where the saved chckpoint is.")

# Flags
parser.add_argument("-gpu", action="store_true",
                    help="Uses the gpu for training. By default it will use the cpu.")
parser.add_argument('-k', "--topk", type=int, help="The top number of probabilities.")
parser.add_argument("--category_names", type=str, help="The list of category names. You need to input the file name with .json as termination.")

args = parser.parse_args()

###############################################################################################################################################
############# MAIN
###############################################################################################################################################

DLmodel, optimizer, criterion, previous_epochs = dlfunctions.load_checkpoint('{}.pth'.format(args.checkpoint_dir))

if args.gpu:
    print("GPU mode activated for prediction")
    device = 'cuda'
else:
    print("CPU mode activated for prediction")
    device = 'cpu'
    
image_path = args.image_data_dir

if args.topk != None:    
    topk = args.topk
else: 
    topk = 5

probs, classes_index = dlfunctions.predict(image_path, DLmodel, topk, device)

if args.category_names != None:    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else: 
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
class_names = []
# Get the names fo the classes
for index in classes_index:

    class_names.append(cat_to_name[index])
    
print('This is the most probable class:',  class_names[0], 'With a probability of ', probs[0]*100, '%')

for i in range(1, len(class_names)):
    print('These are other potential classes and their probabilities:', class_names[i], 'With a probability of ', probs[i]*100, '%')

# python predict.py  flowers/test/20/image_04912.jpg checkpoint  
# python predict.py  flowers/test/20/image_04912.jpg checkpoint  -k 10 --category_names cat_to_name.json
# python predict.py  flowers/test/20/image_04912.jpg checkpoint  -k 10 --category_names cat_to_name.json -gpu
