# Misc imports
import numpy as np
import random as rnd

# These are your other files.
from buffer import ReplayBuffer
from model import Model

# Tensorflow if you're using tensorflow
import tensorflow as tf

class DQNAgent():
    def __init__(
        self,
        input_dims,
        output_dims,
        target_update,
        discount,
        buffer_max_size,
        buffer_size_min,
        batch_size
    ):
        self.output_dims = output_dims
        self.input_dims = input_dims

        self.BUFFER_SIZE_MIN = buffer_size_min
        self.BATCH_SIZE = batch_size
        self.TARGET_UPDATE = target_update
        self.DISCOUNT = discount

        self.model = Model(input_dims, output_dims) # the model we run through the environment
        self.target_model = Model(input_dims, output_dims) # the model that we train
        self.replay_memory = ReplayBuffer(buffer_max_size)

        self.update_target_counter = 0

    # Method for predicting an action 
    def get_action(self, state) -> int:
        return np.argmax(self.model(state.reshape(1,4)))
    
    def learn(self) -> float:
        # We just pass through the learn function if the batch size has not been reached. 
        if self.replay_memory.__len__() < self.BUFFER_SIZE_MIN:
            return

        memories = self.replay_memory.collect_memory(self.BATCH_SIZE)
        states = []
        actions = []
        rewards = []
        next_states = []
        terminateds = []

        # split memories into separate lists
        for memory in memories:
            s, a, r, n, t = memory
            
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n)
            terminateds.append(t)

        # Find our predictions
        input = np.array(states).reshape(self.BATCH_SIZE, 4)
        predictions = self.target_model(input)

        input = np.array(next_states).reshape(self.BATCH_SIZE, 4)
        next_state_predictions = self.target_model(input)
        
        # get max value of next state
        next_state_max_q_values = np.amax(next_state_predictions, 1)

        # Calculate our target
        targets = tf.Variable(predictions).numpy() # change to numpy so we can edit values
        for i in range(self.BATCH_SIZE):
            action = actions[i]
            if terminateds[i]:
                targets[i][action] = -1
            else:
                targets[i][action] = rewards[i] + self.DISCOUNT * next_state_max_q_values[i]

        # backward pass
        history = self.target_model.fit(np.array(states), targets, epochs=3, batch_size=self.BATCH_SIZE, verbose=0)

        self.update_target_counter += 1

        if self.update_target_counter % self.TARGET_UPDATE == 0:
            self.model.set_weights(self.target_model.get_weights())

        return history.history['loss'] 

    def save(self, save_to_path: str) -> None:
        pass

    def load(self, load_path: str) -> None:
        #loaded_target = tf.keras.models.load_model(load_path)
        #loaded_model = tf.keras.models.load_model(load_path)
        pass
