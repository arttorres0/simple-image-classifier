from train_arg_parser import get_input_args
from check_gpu import check_gpu
from data_loader import get_datasets, get_dataloaders
from checkpoint_helper import save_checkpoint, load_model_checkpoint
import sys
from create_model import create_model
from train_and_validate_model import train_and_validate_model
from test_model import test_model
import torch

def main():
    '''
        Creates and train a model for an image folder.
        Uses a pretrained model.
        
        If --resume_training_checkpoint argument is set, it will load a
        model checkpoint and resume training it instead of creating a new model.
        
        Usage:
            python train.py data_directory [--save_dir] <saving_directory> [--saving_name] <saving_name>
                [--arch] <pretrained_model_arch> [--learning_rate] <learning_rate> [--epochs] <training_epochs>
                [--hidden_units1] <hidden_units_1st_hidden_layer> [--hidden_units2] <hidden_units_2nd_hidden_layer>
                [--resume_training_checkpoint] <path_to_trained_model_checkpoint> [--gpu]
                
            If --resume_training_checkpoint argument is set, the following arguments will be ignored:
                --arch, --learning_rate, --hidden_units1, --hidden_units2
                
            Defaults:
                --checkpoint_name : "checkpoint.pth",
                --arch : "vgg19",
                --learning_rate : 0.001,
                --epochs : 3,
                --hidden_units1 : 512,
                --hidden_units2 : 256,
                --resume_training_checkpoint : None,
                --gpu : False
    '''

    #get command line arguments
    args = get_input_args()
    
    #check device
    try:
        device = check_gpu(args.gpu)
    except Exception as e:
        print(e)
        sys.exit(0)
    
    #print selected arguments
    print("Selected arguments:")
    print("\tData Folder = {}".format(args.data_directory))
    print("\tCheckpoint save Folder = {}".format(args.save_dir))
    print("\tCheckpoint save Name = {}".format(args.saving_name))
    print("\tEpochs = {}".format(args.epochs))
    if args.resume_training_checkpoint:
        print("\tTrained model checkpoint = {}".format(args.resume_training_checkpoint))
    else:
        print("\tPretrained model architecture = {}".format(args.arch))
        print("\tTraining learning rate = {}".format(args.learning_rate))
        print("\tNumber of hidden units (1st hidden layer) = {}".format(args.hidden_units1))
        print("\tNumber of hidden units (2nd hidden layer) = {}".format(args.hidden_units2))
    print("\tSelected Device = {}".format(device))
    
    #load data
    train_data, validation_data, test_data = get_datasets(args.data_directory)
    trainloader, validationloader, testloader = get_dataloaders(train_data, validation_data, test_data)
    flower_categories = 102

    already_trained_epochs = 0
    
    #creates or loads a model, depending of argument resume_training_checkpoint
    if args.resume_training_checkpoint:
        print("\nLoading model and resuming training!")
        model, criterion, optimizer, already_trained_epochs = load_model_checkpoint(args.resume_training_checkpoint)
        
        #need to move optimizer tensors to device to keep training,
        #if using an already trained checkpoint
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    if device == "cuda":
                        state[k] = v.cuda()
                    else:
                        state[k] = v.cpu()
    
    else:
        print("\nCreating new model!")
        model, criterion, optimizer = create_model(args.arch, args.hidden_units1, args.hidden_units2, flower_categories, args.learning_rate)
        
    model.to(device)

    #trains and validates model
    train_and_validate_model(device, model, criterion, optimizer, args.epochs, already_trained_epochs, trainloader, validationloader)
    
    #tests model
    test_model(device, model, testloader)

    #saves model checkpoint
    epochs = already_trained_epochs + args.epochs
    save_checkpoint(args.arch, args.hidden_units1, args.hidden_units2, flower_categories,
                    train_data, epochs, args.learning_rate, model, optimizer, args.save_dir, args.saving_name)
    
    print("\nEnd of training script.")

    
if __name__ == "__main__":
    main()