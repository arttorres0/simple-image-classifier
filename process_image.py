from PIL import Image
import numpy as np
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Tensor
    '''
    img = Image.open(image)

    #resize
    min_size = 256
    min_size_over_width = min_size / img.size[0]
    min_size_over_heigth = min_size / img.size[1]
    
    if min_size_over_width <= min_size_over_heigth:
        new_size = (int(img.size[0]*min_size_over_heigth), int(img.size[1]*min_size_over_heigth))
    else:
        new_size = (int(img.size[0]*min_size_over_width), int(img.size[1]*min_size_over_width))

    img = img.resize(new_size, Image.ANTIALIAS)
    
    #center crop
    crop_size = 224
    
    img = img.crop((img.size[0]//2 - crop_size//2,
                    img.size[1]//2 - crop_size//2,
                    img.size[0]//2 + crop_size//2,
                    img.size[1]//2 + crop_size//2))
    
    #convert to numpy array
    np_image = np.array(img)
    
    #change range to 0-1
    np_image = np_image / 255 

    #normalize
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    #transpose
    np_image = np_image.transpose((2, 0, 1))
    
    #add one dimension to change shape from (color, width, height) to (batchsize, color, width, height)
    np_image = np_image[np.newaxis, ...]

    #converts to tensor
    np_image = torch.Tensor(np_image)

    return np_image