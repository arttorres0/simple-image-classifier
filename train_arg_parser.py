import argparse

def get_input_args():
    """
        Uses argparse lib to parse command line arguments from user. Returns parse_args() object
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory',
                       help = 'absolute path of to-be-trained folder')
    
    parser.add_argument('--save_dir', type = str, default = "checkpoints/", 
                        help = 'absolute path of saved checkpoints folder') 
    
    parser.add_argument('--saving_name', type = str, default = "checkpoint.pth", 
                        help = 'saving checkpoint name')
    
    parser.add_argument('--arch', type = str, default = 'vgg19',
                        help = 'pretrained model architecture')
        
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                        help = 'training learning rate')
    
    parser.add_argument('--epochs', type = int, default = 3,
                       help = 'number of training epochs')
        
    parser.add_argument('--hidden_units1', type = int, default = 512,
                        help = 'number of hidden units (1st hidden layer)')
        
    parser.add_argument('--hidden_units2', type = int, default = 256,
                        help = 'number of hidden units (2nd hidden layer)')
    
    parser.add_argument('--resume_training_checkpoint', type = str, default = None,
                        help = 'path to a checkpoint of a trained model, to resume training it')
        
    parser.add_argument('--gpu', action = "store_true", default = False,
                        help = 'activates gpu to increase model performance')
        
    parse_arguments = parser.parse_args()

    return parse_arguments