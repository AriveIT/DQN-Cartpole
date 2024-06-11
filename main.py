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



# Hyperparams
input_dims = 4
output_dims = 2
# likely want to put in some other cool things here like batch size, learning rate, etc. 
episodes = 0

# Global Constants, change these
MAX_EPISODES = 1


if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')
    agnet = DQNAgent(input_dims, output_dims)

    # Make the main game loop.  

    # Total training cycles loop (episode: current training cycle || MAX_EPISODE: Total training iterations)
    while episodes < MAX_EPISODES:
        # resetting for each episode

        time_step = 0 #Steps taken in the particular episode
        rewards = [] # reward list for each episode
        
        #agent.replay_memory.erase_memory()

        # Setting the environment to the intial state at the beginnign of each episode
        observation, info = env.reset() # observation: initial state of the environment || info: any other additional info
        time_step = 0
        done = False #flag to be used in the while loop which tells us if the episode has ended or not.

        # runs until the episode doesnt end/terminate/truncate 
        # Runs to process each time step of the episode
        while not done:
            
            # Get a random action from the environment (gym), ideally through your agent
            action = env.action_space.sample()
            
            # Take the action and observe the result
            # Observation: New state of the environment after the action is taken
            # Reward: 
            # terminated: boolean flag if the episode ended because goal was reached or agent failed
            # truncated: boolean flag indicating if ended due to time limit or other constraint
            # info: additional info by the env
            observation, reward, terminated, trunicated, info = env.step(action)
            
            # Accumulate the reward

            # Check if we lost
            if terminated or trunicated:
                done = True


            # Store our memory

            # learn?
            #agent.learn()
            time_step += 1

            env.render()
        
    # TODO: Check if reward normalization makes sense!
    # agent.save()
    env.close()