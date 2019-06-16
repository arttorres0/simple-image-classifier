# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Data Folder Structure Requirement:
Download project and upload image dataset folder following the pattern below:

    Root
    |
    |__Folder
        |
        |__Test Data Folder
        |
        |__Training Data Folder
        |
        |__Validation Data Folder
    

# Usage:
    Main files:
        a) train.py
            Creates and train a model for an image folder.
            Uses a pretrained model.
            
            If --resume_training_checkpoint argument is set, it will load a
            model checkpoint and resume training it instead of creating a new model.
            
            Usage:
                python train.py data_directory [--save_dir] <saving_directory> [--saving_name] <saving_name>
                    [--arch] <pretrained_model_arch> [--learning_rate] <learning_rate> [--epochs] <training_epochs>
                    [--hidden_units1] <hidden_units_1st_hidden_layer> [--hidden_units2] <hidden_units_2nd_hidden_layer>
                    [--resume_training_checkpoint] <path_to_trained_model_checkpoint> [--gpu]

        b) predict.py
            Predict the class (or classes) of an image using a trained deep learning model.
                    
            Usage:
                python predict.py image checkpoint [--top_k] <top_k_classes>
                    [--category_names] <cat_to_names_file> [--gpu]
