import numpy as np

import torch
# import tensorflow 

# pro tip: the target network and the network network need to have the 
# exact same architecture otherwise you cannot copy the weights between them.


def Model(input_dims, output_dims):
    '''this shouldn't need to be a class. this model should be very simple.
    especially for cartpole, like, 2 dense, an output, and no convolutions 
    should be more than enough.'''
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dims, 128, dtype=torch.float32),
        torch.nn.ReLU(),
        torch.nn.Linear(128, output_dims, dtype=torch.float32),
    )
    return model