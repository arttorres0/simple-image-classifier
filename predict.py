from predict_arg_parser import get_input_args
from check_gpu import check_gpu
from process_image import process_image
from checkpoint_helper import load_model_checkpoint
import torch
import numpy as np
import json, sys

def main():
    '''
        Predict the class (or classes) of an image using a trained deep learning model.
        
        Usage:
            python predict.py image checkpoint [--top_k] <top_k_classes>
                [--category_names] <cat_to_names_file> [--gpu]
                
            Defaults:
                --top_k : 1,
                --category_names : None,
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
    print("\tFile path = {}".format(args.image))
    print("\tNumber of classes to show = {}".format(args.top_k))
    print("\tClass name converter file = {}".format(args.category_names))
    print("\tSelected Device = {}".format(device))

    #process image as array
    processed_image = process_image(args.image)
    processed_image = processed_image.to(device)

    #load model from checkpoint
    model, criterion, optimizer, number_of_performed_epochs = load_model_checkpoint(args.checkpoint)

    #predict
    model.to(device);
    model.eval()

    with torch.no_grad():
        log_ps = model(processed_image)
        ps = torch.exp(log_ps)

        probs, idxs = ps.topk(args.top_k)

        probs = np.round(probs.cpu().numpy().tolist()[0], 8)
        idxs = idxs.cpu().numpy().tolist()[0]

        #inverts class_to_idx
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}

        #changes from idx to class
        classes = [idx_to_class[idx] for idx in idxs]
        
        #loads category to name json converter
        if args.category_names:
            with open('cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)
                
        #get actual flower class
        actual_flower_class = args.image.split('/')[2] #assuming the image_path is 'flowers/test/xxx/flower.jpg'

    print("\nResults:")
    
    if args.category_names:
        actual_flower_label = cat_to_name[actual_flower_class]
        print("\tActual Flower Class: {}; Flower Label: {}".format(actual_flower_class, actual_flower_label))
    else:
        print("\tActual Flower Class: {}".format(actual_flower_class))
        
    print("\tPredictions:")
    
    for i in range(len(classes)):
        if args.category_names:
            print("\t\t{}/{} - Class: {}; Flower Label: {}; Probability: {:.2f}%"
                  .format(i+1, args.top_k, classes[i], cat_to_name[classes[i]], probs[i]*100))
        else:
            print("\t\t{}/{} - Class: {}; Probability: {:.2f}%".format(i+1, args.top_k, classes[i], probs[i]*100))

    print("\nEnd of prediction script.")

    
if __name__ == "__main__":
    main()