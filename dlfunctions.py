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
############# PREPROCESSING
###############################################################################################################################################

def image_pre_processing(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return trainloader, testloader, validloader, train_data

###############################################################################################################################################
############# NETWORK DEFINITION
###############################################################################################################################################

def Network(model, Classifier_input_size, Classifier_output_size, hidden_layers, dropout_probabilities, device):

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # We use LogSoftmax to avoid the floating point memory allocation approximations, inter alia.
    # We build the dictionary in a way that is flexible to include more layers, othrwie it is painful.
    
    hyper_parameters_nn = OrderedDict()
    
    hyper_parameters_nn['fc1'] = nn.Linear(Classifier_input_size, hidden_layers[0])
    hyper_parameters_nn['ReLU1'] = nn.ReLU()
    
    # Now the hidden layers
    for i in range(len(hidden_layers)-1):
        
        hyper_parameters_nn['drop_out' + str(i+2)] = nn.Dropout(p=dropout_probabilities[i])
        hyper_parameters_nn['fc' + str(i+2)] = nn.Linear(hidden_layers[i], hidden_layers[i+1])
        hyper_parameters_nn['ReLU' + str(i+2)] = nn.ReLU()
    
    # Now the output layer
    hyper_parameters_nn['drop_out'+str(len(hidden_layers)+1)] = nn.Dropout(p=dropout_probabilities[-1])
    hyper_parameters_nn['fc'+str(len(hidden_layers)+1)] = nn.Linear(hidden_layers[-1], Classifier_output_size)
    hyper_parameters_nn['output'] = nn.LogSoftmax(dim=1)
    
    classifier = nn.Sequential(hyper_parameters_nn)
    
    # change to cuda
    model.to(device)
    
    model.classifier = classifier
    
    return model

###############################################################################################################################################
############# TRAINING NETWORK
###############################################################################################################################################

def train_model(model, trainloader, validloader, print_every, epochs, criterion, optimizer, device):
    
    # change to cuda
    model.to(device)
    
    running_loss = 0
    steps = 0
    
    # If the optimizer values are loaded, they also need to be pass to the device
    # Ref: If the optimizer values are loaded, they also need to be pass to the device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    with active_session():

        for e in range(epochs):

            # Model in training mode, dropout is on
            model.train()

            running_loss = 0

            for images, labels in trainloader:
                
                steps += 1

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward and backward passes
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:

                    # Model in inference mode, dropout is off
                    model.eval()

                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        
                        total = 0
                        correct = 0
                        valid_loss = 0
                        
                        for data in validloader: 
                            
                            images_valid, labels_valid = data
                            images_valid, labels_valid = images_valid.to(device), labels_valid.to(device)
                            
                            outputs = model(images_valid)
                            _, predicted = torch.max(outputs.data, 1)
                            
                            valid_loss += criterion(outputs, labels_valid).item()
                            total += labels_valid.size(0)
                            correct += (predicted == labels_valid).sum().item()

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                          "Validation Accuracy: {:.3f}".format(correct/total))

                    running_loss = 0

                    # Make sure dropout and grads are on for training
                    model.train()

###############################################################################################################################################
############# VALIDATING NETWORK 
###############################################################################################################################################                    
                 
def validate_model(model, testloader, criterion, device):   
    
    # change to cuda
    model.to(device)
    
    model.eval()                      
    
    with torch.no_grad():
        
        correct = 0
        total = 0
        test_loss = 0
        
        for data in testloader:
            
            images_test, labels_test = data
            images_test, labels_test = images_test.to(device), labels_test.to(device)
            
            outputs = model(images_test)
            _, predicted = torch.max(outputs.data, 1)
            
            test_loss += criterion(outputs, labels_test).item()
            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
      "Test Accuracy: {:.3f}".format(correct/total))
    
# I had a loooot of trouble loading the NN, so I decided to save mostly everytihng because otherwise I need to 
# run the training again as I would not be able to loead the NN properly . Then I will have enough options to try out 
# things with the loafer
def create_checkpoint(model, optimizer, train_data, Classifier_input_size, Classifier_output_size, hidden_layers, epochs, dropout_probabilities, criterion):
       
    checkpoint = {'input_size': Classifier_input_size,
                  'output_size': Classifier_output_size,
                  'hidden_layers': [each for each in hidden_layers],
                  'model': model,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'optimizer': optimizer,
                  'optimizer_state_dict': optimizer.state_dict,
                  'dropout_probabilities': dropout_probabilities,
                  'epochs': epochs,
                  'criterion': criterion}
    
    return checkpoint

def load_checkpoint(filepath):
    
#     REF: https://stackoverflow.com/questions/55759311/runtimeerror-cuda-runtime-error-35-cuda-driver-version-is-insufficient-for
    if torch.cuda.is_available():
        
        map_location = lambda storage, loc: storage.cuda()
        
    else:
        
        map_location='cpu'

    checkpoint = torch.load(filepath, map_location=map_location)
    
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    optimizer = checkpoint['optimizer']
    
    epochs = 0
    epochs = checkpoint['epochs']
    
    criterion = checkpoint['criterion']
    
    return model, optimizer, criterion, epochs

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # This sentence "First, resize the images where the shortest side is 256 pixels, ..." is very confusing.
    # Googling aspect ratio: https://en.wikipedia.org/wiki/Aspect_ratio_(image)
    # Easy concept.
    # However, you shoul be more clear, these are the first thoughts I got:
    # Do I have to resize only the ones with the shortest side of 256? If so, to what size? OR should I crop them?
    # Or resize the images to 256 and then crop them? Or resize only the shortest side to 256 and then crop them?
    # Thus, I will just use the transformations from the beginning as it has the same params and 
    # I think it should give thd sameresilts
    
    image_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    # This is a tensor now. 
    transformed_image = image_transform(image)   
                                    
    return transformed_image

def predict(image_path, model, k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    transformed_image = process_image(image)
    # Ref: https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    transformed_image.unsqueeze_(0)
    
    transformed_image = transformed_image.to(device)
    model = model.to(device)
    
    # TODO: Implement the code to predict the class from an image file  
    model.eval()                      
    
    with torch.no_grad():

        outputs = model(transformed_image)
    
    # To convert the output from natural logarithm to probability, we need to power e to the output of the nn, 
    # as we have used LOGSoftmax
    probabilities = torch.exp(outputs)
    
    # We gather the top classes and probabilities
    topk_outputs = probabilities.topk(k)
    # We will run them through a loop, at least the idx, so we will need to convert each element to arrays
    # We squeeze them to get 1d array for easy iteration
    topk_probabilities = np.array(topk_outputs[0].squeeze(0)) 
    topk_idx = np.array(topk_outputs[1].squeeze(0))
    
    # We get back the dict from the model. The trainloader provides you with a dict thaat maps the classes to the actual
    # index that we are interested in. This is probably because the trainloader is created with unordered folders.
    class_to_idx = model.class_to_idx
    
    # We invert the dictionary: https://kite.com/python/answers/how-to-reverse-a-dictionary-in-python
    idx_to_class = {value : key for (key, value) in class_to_idx.items()}
     
    topk_classes = []

    for index in topk_idx:
        
        topk_classes = np.append(topk_classes, idx_to_class[index])

    return topk_probabilities, topk_classes
