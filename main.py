import os
import copy
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
SEED = 1
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Channel Selection Agent')

#Set the GPU you want to use
num_GPU = 1

import tensorflow as tf
from tensorflow.keras.layers import Dense

print('Tensorflow version: ', tf.__version__)

gpus = tf.config.experimental.list_physical_devices("GPU")
print('Number of GPUs available :', len(gpus))

if num_GPU < len(gpus):
    tf.config.experimental.set_visible_devices(gpus[num_GPU], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[num_GPU], True)
    print('Only GPU number', num_GPU, 'used')


def new_grid(random=False, nb_ch=7, nb_states=5, min_nb=1, max_nb=3, display=False):
    if random == False:
        grid = np.array([[1, 1, 1, 1, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1]])
    else:
        poss_ch = []
        nb_good_ch = np.random.randint(min_nb, max_nb)
        poss_ch.append(np.sort(np.random.choice(np.arange(nb_ch), size=(nb_good_ch), replace=False)))

        for i in range(nb_states - 1):
            new = 0
            while new == 0:
                nb_good_ch = np.random.randint(min_nb, max_nb)
                new_ch = np.sort(np.random.choice(np.arange(nb_ch), size=(nb_good_ch), replace=False))
                for i in range(len(poss_ch)):
                    if np.array_equal(poss_ch[i], new_ch):
                        new -= 1
                if new < 0:
                    new = 0
                else:
                    new = 1

            poss_ch.append(new_ch)

        grid = np.zeros((nb_ch, nb_states))

        for i in range(nb_states):
            for j in range(poss_ch[i].shape[0]):
                grid[poss_ch[i][j], i] = 1

    if display == True:
        plt.figure()
        plt.imshow(np.flip(grid, axis=0), cmap='gray', origin="lower", vmin=0,
                   vmax=1)  # , extent=[0, grid.shape[1], 0, grid.shape[0]])

    return grid.astype(np.float32)


class ChooseEnv():
    def __init__(self):
        self.ch_grid = new_grid()

        self.state = 0
        print('This environment has 7 different channels')

    def run(self, choice):
        reward = self.ch_grid[choice, self.state]

        self.state = (self.state + 1) % 5

        return (reward)

choose_env = ChooseEnv()

class GreedyAgent():
    def __init__(self, nb_channels, epsilon):

        self.nb_channels = nb_channels
        self.epsilon = epsilon

        self.nb_ch_access = np.zeros(shape=(self.nb_channels))
        self.sum_rewards = np.zeros(shape=(self.nb_channels))
        self.q_values = np.zeros(shape=(self.nb_channels))

        self.cumulative_reward = 0

    def choose_action(self):
        # First time ?
        if np.all(self.q_values == 0):
            self.choice = np.random.randint(low=0, high=self.nb_channels)

        # Explore ?
        elif np.random.uniform(size=1) < self.epsilon:
            self.choice = np.random.randint(low=0, high=self.nb_channels)

        # Choose the action that has the best q-value
        else:
            self.choice = np.argmax(self.q_values)

        return self.choice

    def update_q_value(self, reward):

        # Update the q_values
        self.nb_ch_access[self.choice] += 1
        self.sum_rewards[self.choice] += reward
        self.q_values[self.choice] = self.sum_rewards[self.choice] / self.nb_ch_access[self.choice]

        # Update the cumulative reward
        self.cumulative_reward += reward

    def get_cum_reward(self):
        return self.cumulative_reward

    def get_q_values(self):
        return self.q_values

greedy_results = []

# Number of packets we send
len_test = 10000

# Range of epsilons we want to test
epsilon_range = np.linspace(0, 1, 11)
for eps in epsilon_range:

    # Set the same seed for each agent
    np.random.seed(SEED)

    # create an agent with a given epsilon value
    agent_g = GreedyAgent(nb_channels=7, epsilon=eps)

    # Send 10.000 packets
    for i in range(len_test):
        action = agent_g.choose_action()
        reward = choose_env.run(action)
        agent_g.update_q_value(reward)

    # Store cumulative reward for that agent
    greedy_results.append(agent_g.get_cum_reward())

# Plot the cumulative reward as a function of the epsilon value
plt.plot(epsilon_range, greedy_results)
plt.ylim(0, len_test)
plt.xlabel('Epsilon')
plt.ylabel('Cumulative reward after 10.000 time steps');

best_epsilon = epsilon_range[np.argmax(greedy_results)]
print('Best epsilon = ' + str(best_epsilon))

regret_dic = {}

for eps in [0, best_epsilon, 1]:

    regret_dic[str(eps)] = []

    action_history = []

    np.random.seed(SEED)
    agent_g = GreedyAgent(nb_channels=7, epsilon=eps)

    for i in range(len_test):
        action = agent_g.choose_action()
        reward = choose_env.run(action)
        agent_g.update_q_value(reward)

        # Record logs
        action_history.append(action)
        regret_dic[str(eps)].append(agent_g.get_cum_reward())

    # Display the actions
    plt.figure(figsize=(10, 5))
    plt.title(('Epsilon=' + str(eps)))
    plt.ylabel('Action taken')
    plt.ylim(-0.5, 6.5)
    plt.xlabel('Time steps')
    plt.scatter(np.arange(len_test), action_history, s=5)
    plt.show()
    print('Q values after 10.000 times steps: \n' + np.array2string(agent_g.get_q_values(), precision=2))


class UCBAgent():
    def __init__(self, nb_channels):

        self.nb_channels = nb_channels
        self.nb_ch_access = np.zeros(shape=(self.nb_channels))
        self.sum_rewards = np.zeros(shape=(self.nb_channels))
        self.q_values = np.zeros(shape=(self.nb_channels))

        self.ucb = np.zeros(shape=(self.nb_channels))
        self.cumulative_reward = 0
        self.time_step = 0

    def choose_action(self):

        # First time ?
        if self.time_step < self.nb_channels:
            self.choice = self.time_step

        else:
            self.choice = np.argmax(self.ucb)

        return self.choice

    def update_ucb_value(self, reward):

        # Update the UCB values
        self.time_step += 1
        self.nb_ch_access[self.choice] += 1
        self.sum_rewards[self.choice] += reward
        self.q_values[self.choice] = self.sum_rewards[self.choice] / self.nb_ch_access[self.choice]
        self.ucb[self.choice] = self.q_values[self.choice] + np.sqrt(
            2 * np.log(self.time_step) / self.nb_ch_access[self.choice])

        # Update the cumulative reward
        self.cumulative_reward += reward

    def get_cum_reward(self):
        return self.cumulative_reward

    def get_q_values(self):
        return self.q_values


action_history = []
regret_dic['ucb'] = []

# Instantiate the agent
np.random.seed(SEED)
agent_ucb = UCBAgent(nb_channels=7)

# Send 10.000 packets
for i in range(len_test):
    action = agent_ucb.choose_action()
    reward = choose_env.run(action)
    agent_ucb.update_ucb_value(reward)

    # Record logs
    action_history.append(action)
    regret_dic['ucb'].append(agent_ucb.get_cum_reward())

plt.figure(figsize=(10, 5))
plt.title('UCB')
plt.ylabel('Action taken')
plt.ylim(-0.5, 6.5)
plt.xlabel('Time steps')
plt.scatter(np.arange(len_test), action_history, s=5)
print('Q values after 10.000 times steps: \n'+np.array2string(agent_ucb.get_q_values(), precision=2))


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('First 100 time steps')
plt.xlabel('Time steps')
plt.ylabel('cumulative reward')
plt.plot(np.arange(100), regret_dic['0'][:100], label='Greedy, epsilon = 0')
plt.plot(np.arange(100), regret_dic[str(best_epsilon)][:100], label='Greedy, epsion = '+str(best_epsilon))
plt.plot(np.arange(100), regret_dic['1'][:100], label ='Greedy, epsilon = 1')
plt.plot(np.arange(100), regret_dic['ucb'][:100], label='UCB')
plt.legend()


plt.subplot(1, 2, 2)
plt.title('First 10.000 time steps')
plt.xlabel('Time steps')
plt.ylabel('cumulative reward')
plt.plot(np.arange(len_test), regret_dic['0'], label='Greedy, epsilon = 0')
plt.plot(np.arange(len_test), regret_dic[str(best_epsilon)], label='Greedy, epsion = '+str(best_epsilon))
plt.plot(np.arange(len_test), regret_dic['1'], label ='Greedy, epsilon = 1')
plt.plot(np.arange(len_test), regret_dic['ucb'], label='UCB')
plt.legend();