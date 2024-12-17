import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import gym

from DQN import DQNAgent

##################################
# Hyperparameters
##################################

# For learning
input_dims = 4 # size of observation/state
output_dims = 2 # action space
TARGET_UPDATE = 5 # frequency of updating weights (# trains before update)
DISCOUNT = 1 # aka gamma

# Buffer/Memory
BUFFER_MAX_SIZE = 10000
BUFFER_SIZE_MIN = 1000 # minimum length of buffer before training starts
BATCH_SIZE = 512

# Used in main loop
EPSILON_START = 1 # exploit vs explore ratio
EPSILON_MIN = 0.05
MAX_EPISODES = 4000

def get_epsilon_linear_decay(n_episodes, ep_min, ep_start):
    n_episodes = n_episodes % 2000
    a = ep_start - n_episodes * 0.0005
    if a < ep_min: a = ep_min
    if n_episodes > 3750: a = 0
    return a

# fill buffer with memories before training loop starts
def bake_memories(n_memories, env, agent):
    observation, _ = env.reset()

    for _ in range(n_memories):
        action = env.action_space.sample()
        new_observation, reward, terminated, truncated, _ = env.step(action)

        # Store our memory 
        agent.replay_memory.store_memory((observation, action, reward, new_observation, terminated or truncated))
        observation = new_observation

        if terminated or truncated:
            observation, _ = env.reset()

# run model in environment after training
def test_model(n_episodes, env, agent):
    observation, _ = env.reset()
    rewards_over_time = []

    for i in range(n_episodes):
        rewards = 0
        done = False

        while not done:
            action = agent.get_action(observation)
        
            observation, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if terminated or truncated:
                done = True
                rewards_over_time.append(rewards)
                observation, _ = env.reset()

    return rewards_over_time

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode=None) # render_mode = None or "human"
    agent = DQNAgent(
        input_dims,
        output_dims, 
        TARGET_UPDATE,
        DISCOUNT,
        BUFFER_MAX_SIZE,
        BUFFER_SIZE_MIN,
        BATCH_SIZE
    )
    
    bake_memories(1000, env, agent)
    agent.replay_memory.erase_memory()
    epsilon = EPSILON_START
    episodes = 0

    rewards_over_time = []
    epsilon_over_time = []
    loss_over_time = []

    # Main game/training loop
    while episodes < MAX_EPISODES:
        time_step = 0
        rewards = 0
        done = False
        observation, _ = env.reset()

        # Play an episode
        while not done:
            
            # Get action
            random = np.random.rand()
            if random < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_action(observation)
            
            # Take the action and observe the result
            new_observation, reward, terminated, truncated, _ = env.step(action)
            
            # Accumulate the reward
            rewards += reward

            # Store our memory 
            agent.replay_memory.store_memory((observation, action, reward, new_observation, terminated or truncated))
            observation = new_observation

            # Check if we lost
            if terminated or truncated:
                done = True

            time_step += 1
            
        loss = agent.learn()

        # Track values for plotting
        rewards_over_time.append(rewards)
        epsilon_over_time.append(epsilon)
        if loss is not None: loss_over_time.append(loss)
        
        # update count and epsilon
        episodes += 1
        epsilon = get_epsilon_linear_decay(episodes, EPSILON_MIN, EPSILON_START)
    
    
    ############################
    # Visualizations
    ############################

    # test final model
    test_rewards = test_model(500, env, agent)
    env.close()

    # how many times did we train
    print("Times trained: " + str(agent.update_target_counter))
    print("Average reward in testing: " + str(np.mean(test_rewards)))

    # parse memories remaining in buffer for plotting
    n = len(agent.replay_memory)
    s = agent.replay_memory.collect_memory(n - 5)
    positions = []
    velocities = []
    angles = []
    for m in s:
        positions.append(m[0][0])
        velocities.append(m[0][1])
        angles.append(m[0][2])

    # plot data
    fig, axs = plt.subplots(2, 3)
    axs[0][0].set_ylabel("Reward")
    axs[0][0].plot(rewards_over_time)
    
    axs[0][1].set_ylabel("Epsilon")
    axs[0][1].plot(epsilon_over_time)

    axs[1][0].set_ylabel("Loss")
    flattened_loss = [item for sublist in loss_over_time for item in sublist]
    axs[1][0].plot(flattened_loss)

    axs[1][1].set_ylabel("velocities")
    axs[1][1].set_xlabel("positions")
    cb = axs[1][1].scatter(positions, velocities, s=0.3, c=angles, norm=colors.CenteredNorm()) #cmap=plt.cm.get_cmap('RdYlBu')
    plt.colorbar(cb)

    axs[0][2].set_ylabel("Test Reward")
    axs[0][2].plot(test_rewards)

    plt.show()