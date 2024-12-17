from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MeanSquaredError     

def Model(input_dims, output_dims):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dims, activation='relu')) # 4 inputs
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_dims,activation='linear')) # 2 outputs/actions

    model.compile(optimizer=RMSprop(), loss=MeanSquaredError(), metrics=['accuracy'])

    return model