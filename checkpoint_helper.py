import torch
from torch import nn
from torch import optim
from torchvision import models
import os

def save_checkpoint(pretrained_network_arch, hidden_units1, hidden_units2, flower_categories,
                    train_data, epochs, learning_rate, model, optimizer, save_dir, saving_name):
    '''
        Gets model parameters and saves in a checkpoint file
    '''
    checkpoint = {
        'pretrained_network_arch' : pretrained_network_arch,
        'hidden_units1' : hidden_units1,
        'hidden_units2' : hidden_units2,
        'flower_categories' : flower_categories,
        'class_to_idx' : train_data.class_to_idx,
        'number_of_performed_epochs' : epochs,
        'learning_rate' : learning_rate,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }

    #ensures correct file path concatenation
    if save_dir[-1] != "/":
        save_dir = save_dir + "/"
        
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    torch.save(checkpoint, save_dir + saving_name)
    print("Checkpoint saved")

def load_model_checkpoint(checkpoint_path):
    '''
        Reads a checkpoint and returns a model based on it, along with criterion,
        number of performed epochs, and optimizer
    '''
    
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    
    pretrained_network_arch = checkpoint['pretrained_network_arch']

    model = getattr(models, pretrained_network_arch)()
    
    classifier_input_features = model.classifier[0].in_features
    hidden_units1 = checkpoint['hidden_units1']
    hidden_units2 = checkpoint['hidden_units2']
    flower_categories = checkpoint['flower_categories']

    #Model classifier architecture
    classifier = nn.Sequential(nn.Linear(classifier_input_features, hidden_units1),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units1, hidden_units2),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units2, flower_categories),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    criterion = nn.NLLLoss()
    
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    number_of_performed_epochs = checkpoint['number_of_performed_epochs']

    return model, criterion, optimizer, number_of_performed_epochs;