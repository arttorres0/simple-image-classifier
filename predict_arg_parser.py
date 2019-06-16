import argparse

def get_input_args():
    """
        Uses argparse lib to parse command line arguments from user. Returns parse_args() object
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image',
                       help = 'absolute path of to-be-predicted image')

    parser.add_argument('checkpoint',
                       help = 'absolute path of checkpoint file of trained model')
    
    parser.add_argument('--top_k', type = int, default = 1, 
                        help = 'number of classes which the model will calculate probability') 
        
    parser.add_argument('--category_names', type = str, default = None,
                        help = 'json file to map class labels from category to actual name') 
        
    parser.add_argument('--gpu', action = "store_true", default = False,
                        help = 'activates gpu to increase model performance')
        
    parse_arguments = parser.parse_args()

    return parse_arguments