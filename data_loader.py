import torch
from torchvision import datasets, transforms, models

def get_datasets(data_directory):
    '''
        Reads a file path and returns datasets for Training, Validation and Testing data
    '''

    #ensures correct file path concatenation
    if data_directory[-1] != "/":
        data_directory = data_directory + "/"
        
    train_dir = data_directory + 'train'
    valid_dir = data_directory + 'valid'
    test_dir = data_directory + 'test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomGrayscale(p=0.2),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                                          ])

    validation_and_test__transforms = transforms.Compose([transforms.Resize(255),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             [0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                                            ])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_and_test__transforms)
    test_data = datasets.ImageFolder(test_dir, transform = validation_and_test__transforms)
    
    return train_data, validation_data, test_data


def get_dataloaders(train_data, validation_data, test_data):
    '''
        Reads Training, Validation and Testing datasets and returns respective dataloaders
    '''

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 50, shuffle = True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size = 50)
    testloader = torch.utils.data.DataLoader(test_data)
    
    return trainloader, validationloader, testloader