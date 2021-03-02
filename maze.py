"""
Using Q Learning to train an agent to traverse an array finding the #3 while
avoiding #2's which are considered landmines. 

e.g. Here we want to get #1 to go step by step to reach #3. In this 
case it needs to go left, then up 3 times... if instead it goes up on the 
first move, it hits a landmine.

[[0. 0. 0. 3. 0.]
 [2. 0. 0. 0. 0.]
 [0. 2. 0. 0. 0.]
 [0. 0. 0. 0. 2.]
 [0. 0. 0. 0. 1.]]

https://medium.com/analytics-vidhya/a-simple-reinforcement-learning-environment-from-scratch-72c37bb44843
"""
import numpy as np
import pandas as pd

class MazeEnv:
    """
    Important params
    self.x, self.y: x,y coordinates of agent
    self.done: flag of variable which turns True when either 
      (i) episode ends, or (ii) maximum number of steps (200) are acheived
    self.MAX_HOR_VAL, self.MAX_VER_VAL: max horizontal and vertical grid size of env.
    """
    # Constructor for GridWorld_Env Object, i.e. our agent
    def __init__(self, hor, ver):
        self.actions = ["left","up","right","down"]
        self.x = 0
        self.y = 0
        self.MAX_HOR_VAL = hor-1
        self.MAX_VER_VAL = ver-1
        self.done = False
        self.episode_length = 0
        self.state_observation = [self.x, self.y]

    # Reset the agent at the start of each episode
    def reset(self):
        self.done = False
        self.episode_length = 0
        self.x, self.y = 0,0
        self.state_observation = [self.x, self.y]
        return [self.x, self.y]

    # Returns the number of actions in the action set
    def action_space(self):
        return self.actions
    
    # Agent takes step, i.e. take action to interact with the env
    def step(self, action):
        # If agent is at the terminal state, end the episode, set 
        # self.done = True
        if self.state_observation == [
            self.MAX_HOR_VAL, self.MAX_VER_VAL
        ]:
            self.done = True
            return np.array(self.state_observation), self.reward, self.done, self.episode_length
        
        elif self.episode_length > 200:
            self.done = True
            return np.array(self.state_observation), self.reward, self.done, self.episode_length
        
        self.action = action
        self.reward = self.get_reward()
        self.state_observation = self.take_action()
        self.episode_length += 1

        if self.episode_length >= 200:
            self.done = True
        
        return np.array(self.state_observation), self.reward, self.done, self.episode_length
    
    def get_reward(self):
        # If agent tries to run out of the grid, penalize -2
        if  (self.x == 0 and self.action == "left") or \
            (self.x == self.MAX_HOR_VAL and self.action == "right"):
            return -2
        elif (self.y == 0 and self.action == "down") or \
             (self.y == self.MAX_VER_VAL and self.action == "up"):
             return -2
        # If agent reaches a terminal state, reward = 0
        elif ((self.x, self.y) == (self.MAX_HOR_VAL-1, self.MAX_VER_VAL)) and \
             self.action == "right":
             return 0
        elif (self.x, self.y) == (self.MAX_HOR_VAL, self.MAX_VER_VAL-1) and \
             self.action == "up":
             return 0
        # For all other states, penalize agent with -1
        else:
            return -1
    
    # Method to take action, remain in the same box if agent tries
    # to run outside the grid, otherwise move one box in the direction
    # of the action
    def take_action(self):
        if self.x > -1 and self.x <= self.MAX_HOR_VAL:
            # If agent is on edges don't move x coord
            if (self.action == "left" and self.x == 0) or \
                (self.action == "right" and self.x == self.MAX_HOR_VAL):
                self.x = self.x
            #  Left, right moves
            elif self.action == "left":
                self.x -= 1
            elif self.action == "right":
                self.x += 1
            # Otherwise we don't move x
            else:
                self.x = self.x
        if self.y > -1 and self.y <= self.MAX_VER_VAL:
            if (self.action == "down" and self.y == 0) or \
                (self.action == "up" and self.y == self.MAX_VER_VAL):
               self.y = self.y
            elif self.action == "down":
                self.y -= 1
            elif self.action == "up":
                self.y += 1

        return [self.x, self.y]

    # create inner class for the agent?? TODO: create this

# Functions for the agent TODO: Make into agent class
def best_state_action_value(current_state):
    max_val = np.inf * -1
    # from IPython.core.debugger import Tracer; Tracer()() 
    for key in current_state.keys():
        if current_state[key] > max_val:
            max_val = current_state[key]
            best_action = key
    return best_action, max_val

def current_state_to_string(state):
    current_state = ''.join(str(int(e)) for e in state)
    return current_state

def get_all_states_as_strings(MAX_HOR_LENGTH, MAX_VER_LENGTH):
    states = []
    for i in range(MAX_HOR_LENGTH):
        for j in range(MAX_VER_LENGTH):
            tmp = [i,j]
            states.append("".join(str(a) for a in tmp))
    return states

def initialize_Q(MAX_HOR_LENGTH, MAX_VER_LENGTH):
    Q = {}
    states = get_all_states_as_strings(MAX_HOR_LENGTH, MAX_VER_LENGTH)
    for state in states:
        Q[state] = {}
        for i in range(4): #Number of actions = 4
            Q[state][i] = np.random.uniform(-2,2,1)
    return Q

MAX_HOR_LENGTH = 4
MAX_VER_LENGTH = 4
MAX_STATES = MAX_HOR_LENGTH*MAX_HOR_LENGTH
TOTAL_EPISODES = 1000
SIM_RUN = 10
SHOW_EVERY = 10
OBSERVATION_SPACE = 2
LEARNING_RATE = 0.05 # alpha in the literature
DISCOUNT = 0.95 # gamma IN the literature
EPSILON = 0.1
START_EPSILON_DECAYING = 150
END_EPSILON_DECAYING = 600
epsilon_decay_value = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

Summed_reward = []

for sim in range(SIM_RUN):
    EPSILON = 0.1 
    done = False
    
    agent = MazeEnv(MAX_HOR_LENGTH, MAX_VER_LENGTH)
    Q_table = initialize_Q(MAX_HOR_LENGTH, MAX_VER_LENGTH) 

    Total_running_reward = []
    action_space = agent.action_space()
    action_indexes = [i for i in range(len(action_space))]
    cnt = 0
    for episode in range(TOTAL_EPISODES):
        done = False
        current_state = agent.reset()
        cnt += 1
        total_episode_reward = 0
        episode_length = 0

        while not done:
            current_state_str = current_state_to_string(current_state)
            kind_of_selection_ = 'None'

            if np.random.uniform() > EPSILON:
                action, max_qt1 = best_state_action_value(Q_table[current_state_str])
                kind_of_selection_ = 'Greedy'
            else:
                action = np.random.choice(action_indexes)
                max_qt1 = Q_table[current_state_str][action]
                kind_of_selection_ = 'Random'

            next_state, reward, done, episode_length = agent.step(action_space[action])
            total_episode_reward += reward
            Q_table[current_state_str][action] += LEARNING_RATE*(reward + DISCOUNT*max_qt1 - Q_table[current_state_str][action])
            # print(f'current state : {current_state}. Action : {action_space[action]}. Next state: {next_state}. Kind of Sel: {kind_of_selection_}')
            current_state = next_state
            cnt+=1

        Total_running_reward.append(total_episode_reward)
        
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            EPSILON -= epsilon_decay_value
    
    if sim == 0:
        Summed_reward = Total_running_reward
    else: 
        Summed_reward = np.vstack((Summed_reward,Total_running_reward))
        
    if sim % SHOW_EVERY == 0:
        print(sim)

# Displaying average reward

df = pd.DataFrame(Summed_reward)
df
Mean_total_reward = df.mean()
Mean_total_reward

import matplotlib.pyplot as plt

plt.plot(Mean_total_reward)
plt.grid()
plt.title('Mean reward after 50 simulation of 1000 Episode each')
plt.xlabel('Episodes')
plt.ylabel('Rewards / Costs')
plt.show()