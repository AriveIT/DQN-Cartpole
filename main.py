import numpy as np
import random as rnd
import time 
import matplotlib.pyplot as plt
import math
import gym
import pygame

import torch
import tensorflow as tf

from DQN import DQNAgent


# Hyperparams
input_dims = 4
output_dims = 2
# likely want to put in some other cool things here like batch size, learning rate, etc. 
episodes = 0
EPSILON = 0.1 # exploit vs explore ratio

# Global Constants, change these
MAX_EPISODES = 1


if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')
    agent = DQNAgent(input_dims, output_dims)

    # Make the main game loop
    while episodes < MAX_EPISODES:
        time_step = 0
        rewards = []
        #agent.replay_memory.erase_memory()
        observation, info = env.reset()
        time_step = 0
        done = False

        while not done:
            
            # Get action, ideally through your agent
            random = np.random.rand()

            if random > EPSILON:
                action = env.action_space.sample()
            else:
                action = agent.get_action(observation.reshape(1,4))
            
            # Take the action and observe the result
            new_observation, reward, terminated, trunicated, info = env.step(action)
            
            # Accumulate the reward
            rewards.append(reward)

            # Check if we lost
            if terminated or trunicated:
                done = True

            # Store our memory 
            agent.replay_memory.store_memory(observation, action, reward, new_observation)
            observation = new_observation

            # learn?
            #agent.learn()
            time_step += 1

            env.render()
        
        episodes += 1
        if EPSILON < 0.9:
            EPSILON += 0.005
    # TODO: Check if reward normalization makes sense!
    # agent.save()
    env.close()