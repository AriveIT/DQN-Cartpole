import numpy as np

import torch
#import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error 


# pro tip: the target network and the network network need to have the 
# exact same architecture otherwise you cannot copy the weights between them.


def Model():
    '''this shouldn't need to be a class. this model should be very simple.
    especially for cartpole, like, 2 dense, an output, and no convolutions 
    should be more than enough.'''
    model = Sequential()
    model.add(Dense(128, input_dim=4, activation='relu')) # 4 inputs
    model.add(Dense(56, activation='relu'))
    model.add(Dense(2,activation='linear')) # 2 outputs/actions

    model.compile(optimizer=RMSprop(), loss=mean_squared_error, metrics=['accuracy'])

    # model.summary()

    return model