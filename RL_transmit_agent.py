"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import os
import copy
import sys      
import time
import schedule
import threading
from datetime import datetime, timedelta
from timeslot import Timeslot  
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
from gnuradio import gr
import pmt
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Softmax
from tensorflow.keras.layers import Dense

from gym import Env
from gym.spaces import Discrete, Box


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore #conda install pyqt
from PyQt5 import QtWidgets

import abc
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent


HISTORY_BUFFER_LEN = 120
DEFAULT_WINDOW_SIZE = 32
EPISODE_LENGTH = 20
NUMBER_OF_EPISODES = 200

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self,number_of_channels=7,seed = 0, num_GPU=0.0, active_threshold= 0.9, slot_time = 1, window_size = 5, initial_channel = 0, gamma = 1, learning_rate = 0.01, epsilon = 1):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='RL Transmit agent',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.uint16]
        )
        
        self.message_port_register_out(pmt.intern('msg_out'))
        self.message_port_register_in(pmt.intern('msg_in'))
        
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        
        #initailize tensorflow and GPU usage
        print('Tensorflow version: ', tf.__version__)
        gpus = tf.config.experimental.list_physical_devices("GPU")
        print('Number of GPUs available :', len(gpus))
        
        if num_GPU < len(gpus):
        	tf.config.experimental.set_visible_devices(gpus[num_GPU], 'GPU')
        	tf.config.experimental.set_memory_growth(gpus[num_GPU], True)
        	print('Only GPU number', num_GPU, 'used')
        tf.random.set_seed(seed)
	
        #initialize Channel sensing
        self.active_threshold = active_threshold
        self.number_of_channels = number_of_channels
        if (self.number_of_channels+1)%2!=0:
        	print('/!\ Please enter an even number of channels')
        self.channels = np.zeros(self.number_of_channels)
        self.slot_time = slot_time
        self.last_datetime = datetime.now()
        self.window_size = window_size
        self.grid = np.zeros((self.window_size, self.number_of_channels))
        self.initial_channel = initial_channel #change to random
        self.grid_flag = 0
        self.channel_decision = self.initial_channel
        
        self.collision = 0
        self.total_error = 0 
        self.total_count = 1 
        #initialize DQN env
        #self.dq_env = self.DQEnv(self.number_of_channels, self.window_size, False, self.grid.T)

        #initialize DQN Neural Network
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.actions_number = 2
        self.epsilon = epsilon
        self.nb_trainings = 300
        #############################################################        
        self.num_iterations  = 300
        self.initial_collect_steps = 100
        self.replay_buffer_max_length = HISTORY_BUFFER_LEN
        self.batch_size = 64
        self.log_interval = 200
        self.num_eval_episodes = 10
        self.eval_interval = 1000
        #############################################################        
        self.loss = []  # Keep trak of the losses
        self.channel_decision = 0 #wait
        self.train_flag = False
        self.done = False
        self.hist =0
        '''
        self.agent_dq = self.DQAgent(self.number_of_channels, self.learning_rate, self.gamma)
        #sensed_ch, curr_ch, end = self.dq_env.get_init_state()
        #print('##################################')
        #print('ENV initial state:')
        #print('State: ', sensed_ch,', Current Channel: ', curr_ch)
        #print('##################################')
        '''
        
        
        #############################################################
        self.channel = []
        
        self.environment = self.transmit_wait(0,self.window_size,0) ##init state to 0 , window, end to 0
        self.agent = self.DQN_agent(self.learning_rate, self.gamma) #init agent spaces

    def work(self, input_items, output_items):       	
        output_items[0][:]= self.channel_decision
        #Check slot time
        if (datetime.now() - self.last_datetime) > timedelta(days=0, seconds=self.slot_time, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
            	
            	#Sample and mark channel activity
            	active = np.mean(abs(input_items[0][:]))>self.active_threshold
            	self.sensed = active
            	'''
            	if active:
            		self.channel.append(1)
            	else:
            		self.channel.append(0)
            	
            	   
            	self.grid_flag += 1
            	'''
            	#Full window sized grid
            	#if (self.grid_flag == self.window_size+1): 
            	#	self.train_flag = True
            	#	self.channel.pop(0)
            	#	print('history: ',self.channel)
       
		# Run 200 episodes
            	
            	
            	
            	#self.environment._state = [0,0,0]
            	
            	#action = self.agent.choose_action(state, self.epsilon) #choose action based on state (expolit) and epsilon (explore)
            	#reward, new_state, end = self.environment.step(action, int(active))
            	#end = new_state[2]
    
            	#self.agent.insert_history(state, action, reward, new_state) # Insert state, action, reward, new state in history 
    
            	# While it's not the end of the episode
            	
	

            	if self.hist == HISTORY_BUFFER_LEN:
            		self.train_flag = True
            		self.hist += 1
            	
            	self.last_datetime = datetime.now()     
            	
            	if self.done == True:
            		if self.environment._state[2]==0 :
            			self.channel_decision = self.agent.choose_action(self.environment._state, 0)
            			action = self.channel_decision
            			    
            			print ('transmit: ', self.channel_decision, 'state:  ',  self.environment._state   )
            			reward, new_state, end = self.environment.step(action, int(active))
            			self.environment._state = new_state 
            		else:  
            			self.environment.reset()          	
            	
            	else:
            		if self.environment._state[2]==0 :
            			state = self.environment._state
            			print('state:', state)
            			action = self.agent.choose_action(self.environment._state, self.epsilon)
            			self.channel_decision = action
            			reward, new_state, end = self.environment.step(action, int(active))
            			self.environment._state = new_state
            			self.agent.insert_history(state, action, reward, new_state) # Insert state, action, reward, new state in history
            			self.hist += 1
            			print('insert history ', self.hist, 'S',state,'A', action,'R', reward,'SN', new_state)
            		else:
            			self.environment.reset()	       
       
       
            			
        if self.train_flag == True:	
        	print('Replay buffer History populated')
        	#Feed env with initial schannel state
        	self.environment = self.transmit_wait(int(active),self.window_size,0)
        	#start traning  
        	for episode in range(NUMBER_OF_EPISODES):
            		self.environment._state = self.environment.reset()[1]
            		done = False
            		score = 0
            		
            		while not done:
            			action = self.agent.choose_action(self.environment._state, self.epsilon)
            			reward, new_state , end = self.environment.step(action, int(active))
            			
            			self.agent.insert_history(self.environment._state, action, reward, new_state) # Insert S,A,R,S' history 
            			self.agent.learn(16) # Each time we store a new history, we perform a training on random data
            			self.environment._state = new_state
            			score = reward
            			if new_state[2] == 1:
            				done = True
            		print(' Episode:{} Score:{}' .format(episode, score))
        	self.train_flag = False
        	self.done = True
        	print('Tranining done')
	
          
        
        	
        	
        	         
         ## random decision at first (choose threshhold)
            
        '''
            for i in range(0,np.size(input_items,0)):
            	#Energy detected during time slot, mark as occupied 
            	if (np.mean(abs(input_items[i][:]))>self.active_threshold):
            		self.channels[i] = 1
            for j in range (np.size(self.channels)): self.channels[j] = np.abs(self.channels[j]-1)#flip ones and zeros
            print(self.channels,'    ', datetime.now())   
            #when training is done, make a decision
            if self.done == True:
            	state = [self.channels.astype(np.float32), self.dq_env.one_hot(self.channel_decision, self.number_of_channels), 0]
            	self.channel_decision = self.agent_dq.choose_action(state, 0)# epsilon = 0 only expoiltation wehn not training
            	self.total_count += 1
            	self.total_ratio = (self.total_error)/self.total_count
            	print('CH: ',self.channels)
            	temp = np.zeros(np.size(self.channels))
            	temp[self.channel_decision]=1
            	print('DC: ',temp)
            	print('count: ', self.total_count, ' Errors: ', self.total_error, ' ratio: ',self.total_ratio)
         	
            	if self.channels[self.channel_decision] == 0:
            		self.collision += 1             
            self.grid = self.make_grid(self.grid,(self.channels))
            self.last_datetime = datetime.now()
            self.channels = np.zeros(self.number_of_channels)
            #print(self.grid)
            
            self.grid_flag += 1
        output_items[0][:]= self.channel_decision      
        if self.collision == 1:
        	self.total_error += 1
        	self.collision = 0
        	#self.grid_flag = self.window_size+1
        	'''

        	
        '''#Full window sized grid
        if (self.grid_flag == self.window_size+1): 
		
        	#initialize DQN env
        	self.dq_env = self.DQEnv(self.number_of_channels, self.window_size, self.initial_channel, self.grid.T)
	
		# Run 200 episodes
        	for i in range(HISTORY_BUFFER_LEN):
                    
                    self.dq_env.initialize()
                    
                    state = self.dq_env.get_init_state()
                    action = self.agent_dq.choose_action(state, self.epsilon)
                    reward, new_state = self.dq_env.run(action)
	    
                    self.agent_dq.insert_history(state, action, reward, new_state) # Insert state, action, reward, new state in history 
	    
                    # While it's not the end of the episode
                    while new_state[2]==0 :
                    	state = new_state
                    	action = self.agent_dq.choose_action(state, self.epsilon)
                    	reward, new_state = self.dq_env.run(action)
		
                    	self.agent_dq.insert_history(state, action, reward, new_state) # Insert state, action, reward, new state in history
        	
        	self.train_flag = True                      	
        	print('Replay buffer History populated')
        	                 	 

        	
        	

        if self.train_flag == True:
	        print('Training...')
        	self.train_flag = False
        	
        	for i in range(self.nb_trainings):
        		self.dq_env.initialize()
        		state = self.dq_env.get_init_state()
    
        		action = self.agent_dq.choose_action(state, self.epsilon)
        		reward, new_state = self.dq_env.run(action)
        		self.agent_dq.insert_history(state, action, reward, new_state) # Insert state, action, reward, new state in history 
        		self.agent_dq.learn(64) # Each time we store a new history, we perform a training on random data
    
        		# While it's not the end of the episode
        		while new_state[2]==0 :
        			state = new_state
        			action = self.agent_dq.choose_action(state, self.epsilon)
        			reward, new_state = self.dq_env.run(action)
        			self.agent_dq.insert_history(state, action, reward, new_state) # Insert state, action, reward, new state in history 
        			self.agent_dq.learn(64) # Each time we store a new history, we perform a training on random data
    
        		self.loss.append(tf.reduce_mean(self.agent_dq.loss).numpy()) #Save the losses for future visualization
        			#Every 10 iterations we copy the parameters of the online DQN to the offline DQN
        		if (i+1)%10 == 0:
        			self.agent_dq.copy_parameters()
        			print((i+1), einitial_stnd=', ')
        	self.done = True
        	#plt.semilogy(np.arange(len(self.loss)), self.loss)
        	#plt.show()
        	self.evaluate_dq_agent(self.agent_dq , self.dq_env.grid)
        '''
        return len(output_items[0])
        

    class transmit_wait():

    	def __init__(self, sensed = 0 , window_size = DEFAULT_WINDOW_SIZE, initial_state = 0 ):
    		self._action_spec = Discrete(2) #transmit or not {0,1}
    		self._observation_spec = Discrete(2) #occupied or not {0,1}
    		self.initial_state = initial_state #no transmision
    		#self.window_size = window_size
    		self.sensed = sensed
    		self._state = [initial_state , sensed , 0]
    		self.episode_length = EPISODE_LENGTH
    		self.reward = 0
    		
    	def action_spec(self):
    		return self._action_spec

    	def observation_spec(self):
    		return self._observation_spec

    	def reset(self):
    		self._state = [self.initial_state , self.sensed , 0]
    		self._episode_ended = False
    		self.episode_length = EPISODE_LENGTH
    		self.reward = 0
    		return (self.reward , self._state , 0)

    	def step(self, action, sensed):
    		prv_action, prv_sensed, self._episode_ended = self._state
    		'''
    		if self._episode_ended:
    			# The last action ended the episode. Ignore the current action and start
    			# a new episode.
    			self.reset()
    		'''	

    		self._state = [action , sensed , 0] #new stat
    		#print('Gstate_step:', self._state)
    		if prv_sensed == 1 and prv_action == 1: #collision
    			self.reward = self.reward-2
    		elif prv_sensed == 0 and prv_action == 1: #clean transmit
    			self.reward = self.reward+3
    		elif prv_sensed == 1 and prv_action == 0: #avoided collision
    			self.reward = self.reward+1
    		elif prv_sensed == 0 and prv_action == 0: #wasted slot
    			self.reward = self.reward-1	
    		reward = self.reward	#comulative reward
    		
    		#episode done?	(number of slots passed in an episode)
    		self.episode_length  = self.episode_length - 1
    		if self.episode_length <= 0:
    			self._state = [action , sensed , 1] #new state
    		#print('prv action',prv_action,'prv_sensed', prv_sensed,'end?' ,self._episode_ended)
    		#print('action', action, 'sensed', sensed)
    		# Make sure episodes don't go on forever.
    		if self._episode_ended: 
    			return(reward, self._state , 1)
    		else:
    			return(reward, self._state , 0)
			    			
    class DQN_agent():
    	def __init__(self, learning_rate, gamma):
    		self.nb_actions = 2
    		self.gamma = gamma
    		self.learning_rate = learning_rate
    		
    		self.history_length = HISTORY_BUFFER_LEN
    		self.history = [[]for i in range(self.history_length)]
    		self.history_idx = 0
    		
    		#Create and initialize the online DQN
    		self.DQN_online = tf.keras.models.Sequential([Dense(2, activation='relu'), Dense(self.nb_actions, activation='softplus') #Outputs positive values 
    		])
    		self.DQN_online.build(input_shape=(None, 2)) #Build the model to create the weights
    		
    		#Create and initialize the offline DQN
    		self.DQN_offline = tf.keras.models.Sequential([Dense(2, activation='relu'), Dense(self.nb_actions, activation='softplus') #Outputs positive values
    		])
    		self.DQN_offline.build(input_shape=(None, 2)) #Build the model to create the weights
    		
    		self.copy_parameters() #Copy the weights of the online network to the offline network
    		
    		self.loss_func = tf.keras.losses.MSE
    		self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        
    	def choose_action(self, state, epsilon):
    		"""Implements an epsilon-greedy policy"""
    		#sensed_ch, curr_ch, end = state
    		transmit , sensed , end = state
    		# Explore ?
    		if np.random.uniform(size=1) < epsilon :
    			action =  np.random.randint(2) ####### Transmit with probability epsilon
    			print('random action chosen:    ', action)

    		#Exploite - Choose the current best action
    		else:    
    			DQN_input = tf.concat([[int(transmit)], [int(sensed)]], axis=0)[tf.newaxis, :] #Create a state vector, which is the DQN input. Shape : [1, 1]
    			outputs = self.DQN_online(DQN_input).numpy() #Get the predicted Q values corresponding to the 2 actions
    			action = np.argmax(outputs) #Take the action that has the highest predicted Q value (0, 1)
    			print('non random action chosen:    ', action)			
    		return action
    
    	def learn(self, batch_size):
    		"""Sample experiences from the history and performs SGD"""
    		
    		# Samples random experiences from the history
    		idx = np.random.choice(range(self.history_length), batch_size, replace=False) # Create random indexes 
    		rdm_exp =  [self.history[i] for i in idx] # Take experiences corresponding to the random indexes
    		
    		# Each experience is written in this format : [state_vec, end, action, reward, n_state_vec, n_end] (see insert_history method)
    		
    		# Create 6 batches : states_vec, end_boolean, actions, rewards, new states_vec, new_end_boolean
    		states_vec = np.array([rdm_exp[i][0] for i in range(batch_size)]) # Shape : [Bs, 2*self.nb_ch]
    		end = np.array([rdm_exp[i][1] for i in range(batch_size)]) # Shape : [Bs]
    		actions = np.array([rdm_exp[i][2] for i in range(batch_size)]) # Shape : [BS]
    		rewards = np.array([rdm_exp[i][3] for i in range(batch_size)]) # Shape : [BS]
    		n_states_vec = np.array([rdm_exp[i][4] for i in range(batch_size)]) # Shape : [Bs, 2*self.nb_ch]
    		n_end = np.array([rdm_exp[i][5] for i in range(batch_size)]) # Shape : [BS]
    		
    		#Compute the best q_value for the new states
    		max_n_q_values = tf.reduce_max(self.DQN_offline(n_states_vec), axis=1).numpy()

    		with tf.GradientTape() as tape:
    			#Forward pass through the online network to predict the q_values
    			pred_q_values = self.DQN_online(states_vec)
    			
    			# Compute targets
    			targets = pred_q_values.numpy()
    			targets[np.arange(targets.shape[0]), actions]= rewards + (1-n_end) * self.gamma * max_n_q_values
    			
    			# Evaluate the loss
    			self.loss = self.loss_func(pred_q_values, targets)
        
    		# Compute gradients and perform the gradient descent
    		gradients = tape.gradient(self.loss, self.DQN_online.trainable_weights)
    		self.optimizer.apply_gradients(zip(gradients, self.DQN_online.trainable_weights))  
    
    	def insert_history(self, state, action, reward, n_state):
    		"""Insert experience in history"""
        
    		#sensed_ch, curr_ch, end = state
    		transmit , sensed , end = state
    		state_vec = np.array([int(transmit),int(sensed)])# Create the state vector for the state
        
    		n_transmit, n_sensed, n_end = n_state
    		n_state_vec = np.array([n_transmit, n_sensed]) # Create the state vector for the new state

    		self.history[self.history_idx] = [state_vec, end, action, reward, n_state_vec, n_end] # Insert everything in the history
        
    		self.history_idx = (self.history_idx+1)%self.history_length # Move the history_idx by one
    
    	def copy_parameters(self):
    		"""Copy the parameters of the online network to the offline network"""

    		weights = self.DQN_online.get_weights()
    		self.DQN_offline.set_weights(weights)			
''' 
    class DQEnv():
    	def __init__(self, nb_channels = 7, nb_states=5, initial_state = 0 , grid = []):
        
	    	if (nb_channels+1)%2!=0:
	    		print('/!\ Please enter an even number of channels')
            
	    	self.nb_ch = nb_channels
	    	self.nb_states = nb_states
        
	    	self.grid = grid
	    	print ('env grid:')
	    	print(self.grid)
	    	self.init_state = [int(initial_state), int(self.nb_states-1)]
	    	self.ch_state = self.init_state[1]
	    	self.curr_ch = self.init_state[0]
	    	self.sent_mess = 0
        
	    	print('This environment has ' + str(self.nb_ch) +' different channels')
        
    	def run(self, action):
        
	    	self.sent_mess += 1
        
	    	self.ch_state  = (self.ch_state + 1)%self.nb_states
        
	    	self.curr_ch = action
	    	
	    	reward = np.abs(self.grid[self.curr_ch, self.ch_state])
        
	    	if self.sent_mess != self.nb_states: 
	    		end = 0
	    	else :
	    		end = 1
        
	    	return(reward, [self.grid[:, self.ch_state], self.one_hot(self.curr_ch, self.nb_ch), end])
    
    	def get_init_state(self):
	    	return [self.grid[:, self.init_state[1]], self.one_hot(self.init_state[0], self.nb_ch), 0]
    
    	def initialize(self):
	    	self.ch_state = self.init_state[1]
	    	self.curr_ch = self.init_state[0]
	    	self.sent_mess = 0
        
    	def one_hot(self, index, depth):
	    	oh = np.zeros(depth, dtype=np.float32)
	    	oh[index] = 1
	    	return oh
    
    
    class DQAgent():
    	def __init__(self, nb_channels, learning_rate, gamma):
    		self.nb_ch = nb_channels
    		self.nb_actions = nb_channels
    		self.gamma = gamma
    		self.learning_rate = learning_rate
    		
    		self.history_length = HISTORY_BUFFER_LEN
    		self.history = [[]for i in range(self.history_length)]
    		self.history_idx = 0
    		
    		#Create and initialize the online DQN
    		self.DQN_online = tf.keras.models.Sequential([Dense(2*self.nb_ch, activation='relu'), Dense(self.nb_actions, activation='softplus') #Outputs positive values 
    		])
    		self.DQN_online.build(input_shape=(None, self.nb_ch*2)) #Build the model to create the weights
    		
    		#Create and initialize the offline DQN
    		self.DQN_offline = tf.keras.models.Sequential([Dense(2*self.nb_ch, activation='relu'), Dense(self.nb_actions, activation='softplus') #Outputs positive values
    		])
    		self.DQN_offline.build(input_shape=(None, self.nb_ch*2)) #Build the model to create the weights
    		
    		self.copy_parameters() #Copy the weights of the online network to the offline network
    		
    		self.loss_func = tf.keras.losses.MSE
    		self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        
    	def choose_action(self, state, epsilon):
    		"""Implements an epsilon-greedy policy"""
    		sensed_ch, curr_ch, end = state
    		# Explore ?
    		if np.random.uniform(size=1) < epsilon :
    			action =  np.random.randint(low = 0, high = np.size(sensed_ch)) ####### Take one action randomly {0, 1, 2, 3, 4, 5, 6} with probability epsilon
    			
        
    		#Choose the best action
    		else:    
    			#sensed_ch, curr_ch, end = state #Decomposes the state

    			DQN_input = tf.concat([sensed_ch, curr_ch], axis=0)[tf.newaxis, :] #Create a state vector, which is the DQN input. Shape : [1, 2*self.nb_ch]
    			outputs = self.DQN_online(DQN_input).numpy() #Get the predicted Q values corresponding to the 3 actions
    			action = np.argmax(outputs)#-1 #Take the action that has the highest predicted Q value (0, 1, 2, 3, 4, 5, 6)			
    			#print(outputs)
    		return action
    
    	def learn(self, batch_size):
    		"""Sample experiences from the history and performs SGD"""
    		
    		# Samples random experiences from the history
    		idx = np.random.choice(range(self.history_length), batch_size, replace=False) # Create random indexes 
    		rdm_exp =  [self.history[i] for i in idx] # Take experiences corresponding to the random indexes
    		
    		# Each experience is written in this format : [state_vec, end, action, reward, n_state_vec, n_end] (see insert_history method)
    		
    		# Create 6 batches : states_vec, end_boolean, actions, rewards, new states_vec, new_end_boolean
    		states_vec = np.array([rdm_exp[i][0] for i in range(batch_size)]) # Shape : [Bs, 2*self.nb_ch]
    		end = np.array([rdm_exp[i][1] for i in range(batch_size)]) # Shape : [Bs]
    		actions = np.array([rdm_exp[i][2] for i in range(batch_size)]) # Shape : [BS]
    		rewards = np.array([rdm_exp[i][3] for i in range(batch_size)]) # Shape : [BS]
    		n_states_vec = np.array([rdm_exp[i][4] for i in range(batch_size)]) # Shape : [Bs, 2*self.nb_ch]
    		n_end = np.array([rdm_exp[i][5] for i in range(batch_size)]) # Shape : [BS]
    		
    		#Compute the best q_value for the new states
    		max_n_q_values = tf.reduce_max(self.DQN_offline(n_states_vec), axis=1).numpy()

    		with tf.GradientTape() as tape:
    			#Forward pass through the online network to predict the q_values
    			pred_q_values = self.DQN_online(states_vec)
    			
    			# Compute targets
    			targets = pred_q_values.numpy()
    			targets[np.arange(targets.shape[0]), actions]= rewards + (1-n_end) * self.gamma * max_n_q_values
    			
    			# Evaluate the loss
    			self.loss = self.loss_func(pred_q_values, targets)
        
    		# Compute gradients and perform the gradient descent
    		gradients = tape.gradient(self.loss, self.DQN_online.trainable_weights)
    		self.optimizer.apply_gradients(zip(gradients, self.DQN_online.trainable_weights))  
    
    	def insert_history(self, state, action, reward, n_state):
    		"""Insert experience in history"""
        
    		sensed_ch, curr_ch, end = state
    		state_vec = np.concatenate([sensed_ch, curr_ch], axis=0) # Create the state vector for the state
        
    		n_sensed_ch, n_curr_ch, n_end = n_state
    		n_state_vec = np.concatenate([n_sensed_ch, n_curr_ch], axis=0) # Create the state vector for the new state

    		self.history[self.history_idx] = [state_vec, end, action, reward, n_state_vec, n_end] # Insert everything in the history
        
    		self.history_idx = (self.history_idx+1)%self.history_length # Move the history_idx by one
    
    	def copy_parameters(self):
    		"""Copy the parameters of the online network to the offline network"""

    		weights = self.DQN_online.get_weights()
    		self.DQN_offline.set_weights(weights)

		

    def make_grid(self,grid,channels):
        grid = np.append(grid, channels).reshape(self.window_size+1,7)
        grid = np.delete(grid, 0, 0)
        grid = grid.reshape(self.window_size,self.number_of_channels)
        return grid.astype(np.float32)
  
    def evaluate_dq_agent(self, agent, grid):

        action_history=[]
        tot_reward = 0
    
        self.dq_env.initialize()
        first_state = self.dq_env.get_init_state()
        action = agent.choose_action(first_state, epsilon = 0)
        action_history.append(action)
        reward, new_state = self.dq_env.run(action)
        tot_reward += reward
    
        for j in range(grid.shape[1]-1):
            state = new_state
            action = agent.choose_action(state, epsilon = 0)
            action_history.append(action)
            reward, new_state = self.dq_env.run(action)
            tot_reward += reward
            
        choosen_channels = [(grid.shape[0]-1)/2]
        for i in range(len(action_history)):
        	choosen_channels.append((choosen_channels[i]+action_history[i])%grid.shape[0])
        choosen_channels = choosen_channels[1:]
        #plt.imshow(np.flip(grid, axis=0), origin="lower", cmap='gray', vmin=0, vmax=1)
        #for i in range(len(choosen_channels)):
        #	plt.scatter(i, grid.shape[0]-1-choosen_channels[i], color='r')
        #plt.show()
        #print(str(int(tot_reward))+'/'+str(grid.shape[1])+' packets have been transmitted')
        return tot_reward
        '''  
