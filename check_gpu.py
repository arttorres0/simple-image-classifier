import torch

def check_gpu(gpu):
    '''
    Checks if user wants to use GPU or CPU and returns the device string.
    Returns exception if GPU is selected but it's not available.
    '''
    if gpu:
        if torch.cuda.is_available():
            return "cuda"
        else:
            raise Exception("Exception: You selected GPU, but it is not availabe in this workspace")
    else:
        return "cpu"