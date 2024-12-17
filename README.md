# Deep Q-Network (DQN) Implementation for CartPole

This project implements a **Deep Q-Network (DQN)** using **TensorFlow** to solve the classic reinforcement learning environment **CartPole-v1** from OpenAI Gym.
The CartPole problem requires an agent to balance a pole on a cart by applying forces to the left or right.

# Files
buffer.py: store memories, consisting of a (state, action, reward, next state, terminated) tuple  
DQN.py: learn function  
main.py: set up, and train loop  
model.py: model initialization

# Hyperparameters
**For learning** 
```
input_dims = 4 # size of observation/state  
output_dims = 2 # action space  
TARGET_UPDATE = 5 # frequency of updating weights (# trains before updating target model)  
DISCOUNT = 1 # aka gamma 
```

**Buffer/Memory**   
```
BUFFER_MAX_SIZE = 10000  
BUFFER_SIZE_MIN = 1000 # minimum length of buffer before training starts  
BATCH_SIZE = 512 
```

**Used in main loop**  
``` 
EPSILON_START = 1 # exploit vs explore ratio  
EPSILON_MIN = 0.05  
MAX_EPISODES = 4000 
```

# Results
After training, the agent was able to balance the pole upright for **500 time steps**
