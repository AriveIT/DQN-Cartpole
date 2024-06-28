import numpy as np
import random as rnd
import time 
import matplotlib.pyplot as plt
import math
import gym
import pygame

import torch
# import tensorflow as tf

# all of the libraries above can be installed with pip
# ex: pip install numpy or pip install torch

from DQN import DQNAgent
from buffer import ReplayBuffer

# Hyperparams
input_dims = 4
output_dims = 2
# likely want to put in some other cool things here like batch size, learning rate, etc. 

# Global Constants, change these
MAX_EPISODES = 2000
episodes = 0
Episodes = np.zeros(MAX_EPISODES)
RewardList = []

if __name__ == "__main__":
    #env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v0')
    agent = DQNAgent(input_dims, output_dims)

    # Make the main game loop.  

    while episodes < MAX_EPISODES:
        rewards = []

        observation, info = env.reset()
        done = False
        while not done:
            
            # Get action, ideally through your agent
            action = agent.get_action(observation)
            prev_observation = observation
            # Take the action and observe the result
            observation, reward, terminated, trunicated, info = env.step(action)
            
            # Accumulate the reward
            rewards.append(reward)

            # Check if we lost
            if terminated or trunicated:
                done = True


            # Store our memory
            agent.replay_memory.store_memory((torch.tensor(prev_observation), action, reward, torch.tensor(observation)))

            # learn?
            agent.learn()

            env.render()
        Episodes[episodes] = episodes
        episodes += 1
        RewardList.append(sum(rewards))
    plt.plot(Episodes, RewardList)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()
        
    # TODO: Check if reward normalization makes sense!
    agent.save('savedData.txt')
    env.close()