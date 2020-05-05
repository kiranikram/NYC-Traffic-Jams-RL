import gym
from gym import spaces
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod
#from gym_pull.envs.registration import registry, register, make, spec
import numpy as np
from gym.utils import seeding
from PIL import Image
from skimage.draw import rectangle
import pandas as pd
from collections import namedtuple
from custom_env import NycMaze1
import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
from IPython.display import clear_output
from IPython.display import display
from time import sleep
import matplotlib.image as image
from gym import Env, spaces
from gym.utils import seeding
from matplotlib import animation

#Actions = {'North': 0, 'South': 1, 'East': 2, 'West': 3}

class environment(Env):
  def __init__(self, nS, nA, P, isd, start_state):
    # nS = number of States
    # na = Number of actions
    # P = transitions 
    # isd = initial state distribution 
    self.start_state = start_state
    self.s = start_state 
    self.P = P
    self.isd = isd
    self.lastaction = None
    self.nS = nS
    self.nA = nA

    self.action_space = spaces.Discrete(self.nA)
    self.observation_space = spaces.Discrete(self.nS)
    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    raise NotImplementedError

  def step(self, a):
    raise NotImplementedError

"""## This is our environment class of our nyc maze
A state with two zeros at the end is when traffic is not present, a state with the values of the state copied at the end (2323) is when traffic is present
"""

class driver(environment):
  def __init__(self, **kwargs):
    self.start = np.array([2,2])
    self.maze = None

    self.start = kwargs['start_car']
    self.goal = kwargs['destination']
    self.width = kwargs['block_width']
    self.height = kwargs['block_height']
    self.blocks = kwargs['blocks']
    self.obstacle_width = kwargs['obstacle_width']
    self.obstacle_height = kwargs['obstacle_height']
    self.traffic_reward = kwargs['traffic_reward']
    self.traffics = kwargs['traffics'] #traffic states that are possible can be changed
    self.maze = self.build_maze(self.width,self.height, self.obstacle_width, self.obstacle_height, self.blocks)
    self.set_start(self.start)
    self.state = self.encode_state(self.start[0], self.start[1], 0, 0) #just as a start definition no traffic present 
    self.set_goal(self.goal)
    self.trafic_R = None
    self.trafic_ON = False
    num_states = self.width*self.height*self.width*self.height #needs to be size 10*10*10*10 because that many states available
    num_actions = 4 #"(N,S,E,W)"
    self.observation_space = num_states
    self.action_space = num_actions
    
    num_rows = self.width
    num_col = self.height
    traf_rows = self.width
    traf_cols = self.height
    max_rows = num_rows - 1
    max_col = num_col - 1

    self.R = {state: {action: [] for action in range(num_actions)} for state in range(num_states+1)}

    initial_state_distribution = np.zeros(num_states)
    
    for row in range(num_rows):
      for col in range(num_col):
        for traf_row in range(traf_rows):
          for traf_col in range(traf_cols):
            state = self.encode_state(row,col,traf_row,traf_col)
            if (row,col) != self.goal:
              initial_state_distribution[state] += 1
            for action in range(num_actions):
              new_row, new_col = row, col 
              new_traf_row, new_traf_col = traf_row, traf_col
              done = False
              reward = -1
              car_loc = (row,col)
              if traf_row !=0 and traf_col != 0: #then not in trafic
                #Actions = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
                if action == 0:
                  new_row = max(row - 1, 0)
                  new_traf_row = new_row
                elif action == 1:
                  new_row = min(row + 1, max_rows)
                  new_traf_row = new_row
                elif action == 2:
                  new_col = min(col + 1, max_col)
                  new_traf_col = new_col
                elif action == 3:
                  new_col = max(col - 1, 0)
                  new_traf_col = new_col

                if (new_row,new_col) == (self.goal[0], self.goal[1]):
                  done = True
                  reward = 100 # reached the destination 🏁
                elif self.maze[new_row, new_col] == 1:
                  reward = -200 # this is a crash 🚒🚑
                if self.encode_coord(new_traf_row, new_traf_col) in self.traffics: #if the next state is in predefined state where traffic is then reward set 
                #to predefinded reward
                  reward = self.traffic_reward

                new_state = self.encode_state(new_row,new_col,new_traf_row,new_traf_col)
                self.R[state][action].append(1)
                self.R[state][action].append(new_state)
                self.R[state][action].append(reward)
                self.R[state][action].append(done)
              
              else:
                #Actions = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
                if action == 0:
                  new_row = max(row - 1, 0)
                elif action == 1:
                  new_row = min(row + 1, max_rows)
                elif action == 2:
                  new_col = min(col + 1, max_col)
                elif action == 3:
                  new_col = max(col - 1, 0)

                if (new_row,new_col) == (self.goal[0], self.goal[1]):
                  done = True
                  reward = 100 # reached the destination 🏁
                elif self.maze[new_row, new_col] == 1:
                  reward = -200 # this is a crash 🚒🚑
                
                new_state = self.encode_state(new_row,new_col,0,0)
                self.R[state][action].append(1)
                self.R[state][action].append(new_state)
                self.R[state][action].append(reward)
                self.R[state][action].append(done)
    
    initial_state_distribution /= initial_state_distribution.sum()
    environment.__init__(self, num_states, num_actions, self.R, initial_state_distribution, self.start) 

  def build_maze(self,width, height, obstacle_width, obstacle_height, blocks):
    x = np.ones([height, width], dtype=np.uint8)
    start = (2,2)
    end = (height-3,width -3)
    rr, cc = rectangle(start, end, shape = x.shape)
    x[rr, cc] = 1
    #print(width // blocks)
    for i in range(0,width-1,(width//blocks)):
      x[:,i] = 0
    for i in range(0,height-1,(height//blocks)):
      x[i,:] = 0
    x[1, :] = 0
    x[-2, :] = 0
    x[:, 1] = 0
    x[:, -2] = 0
    #builds the walls
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1
    return x

  def encode_coord(self,row, column):
    i = row
    i *= self.height
    i += column
    return i 

  def encode_state(self,row, column, row_traffic, column_traffic):
    i = row
    i *= self.height
    i += column
    i *= self.width
    i += row_traffic
    i *= self.height
    i += column_traffic
    return i 

  def decode_state(self, state):
    # out = [row,column]
    out = []
    out.append(state % self.width)
    state = state // self.width
    out.append(state % self.height)
    state = state // self.height
    out.append(state % self.width)
    state = state // self.width
    out.append(state % self.height)
    return tuple(reversed(out))

  def decode_coord(self, coord):
    # out = [row,column]
    out = []
    out.append(coord % self.width)
    coord = coord // self.width
    out.append(coord % self.height)
    return tuple(reversed(out))

  def set_start(self, start):
    self.maze[start[0],start[1]] = 10
    self.start = start
    self.state = self.encode_state(start[0], start[1], start[2], start[3])

  def set_goal(self,goal):
    self.maze[goal] = 12
    self.goal = (goal[0], goal[1],0,0)

  def reset(self):
    self.state = self.encode_state(self.start[0], self.start[1],0,0)
    return self.state

  def step(self, a):
    x,y = self.decode_state(self.state)[0], self.decode_state(self.state)[1]
    if self.trafic_ON:
      env.state = self.encode_state(x,y,x,y)
    if self.trafic_ON is False:
      env.state = self.encode_state(x,y,0,0)
    transitions = self.R[self.state][a]
    prob, state, reward, done = transitions[0], transitions[1], transitions[2], transitions[3]
    return prob, state, reward, done

  def render(self, states = []):
    out = self.maze.copy()
    out[np.where(out == 0)] = 2
    out[np.where(out == 1)] = 4
    if self.trafic_ON:
      for i in range(len(self.traffics)):
        out[self.decode_coord(traffics[i])] = 9
    for i in range(len(states)):
      (x,y,_,_) = self.decode_state(states[i])
      out[x,y] = 5
    #print("decoding state: ", self.state)
    car_row, car_col,_,_ = self.decode_state(self.state)
    #print("car_row: ", car_row, "car_col: ", car_col)
    out[car_row][car_col] = 7
    img = out
    return img

# size of the grid
block_width = 10 
block_height = 10
blocks = 3
start_car = ((block_height-2),1,0,0)
destination = (1, block_width-2)
#where possible traffic jams can occur
traffics = [61, 62, 63, 64, 65, 66, 67, 23, 26, 28] 

environment_params = {'block_width': block_width, 'block_height': block_height, 'blocks' : blocks, 'obstacle_width': block_width - 2, 'obstacle_height': 3, 'start_car': start_car, 'destination': destination, 'traffic_reward': -15, 'traffics':traffics}
env = driver(**environment_params)
print("start:",env.start,"destination:",env.goal)
print(env.maze)
print("Action Space '{}'".format(env.action_space))
print("State Space '{}'".format(env.observation_space))

print("state", env.state)
env.trafic_ON = True
img = env.render()
plt.imshow(img, cmap= 'viridis')

if env.start == env.state:
  print("we are in start position state:", env.state)
else:
  print("state:", env.state)

fig, axs = plt.subplots(1,5, figsize = (15,15))
transitions_val = []

env.trafic_ON = False
#we set a state where we are next to the goal
print("\n setting state to state next to goal")
state = (destination[0] + 1, destination[1])
state = env.encode_state(state[0],state[1],0,0)
env.state = state
if env.start == env.state:
  print("we are in start position state:", env.state)
else:
  print("state:", env.state)

#print("destination:",env.goal)
#print(env.R[env.state])
transitions_val.append(env.R[env.state])
#print("transitions: ", env.R[env.state][1])
img = env.render()
axs[0].imshow(img)
axs[0].set_title("State next to the goal")

#we set a state at a intersection to check 
print("\n setting state to state at intersection")
intersect = block_width / blocks
intersect = int(intersect)
state = env.encode_state(intersect,intersect,0,0)
env.state = state
if env.start == env.state:
  print("we are in start position state:", env.state)
else:
  print("state:", env.state)
transitions_val.append(env.R[env.state])
#print("destination:",env.goal)
#print(env.R[36])
img = env.render()
axs[1].imshow(img)
axs[1].set_title("State at intersection")

#we set a state at a intersection with traffic true
print("\n setting state to state at intersection with traffic true")
env.trafic_ON = True
intersect = block_width / blocks
intersect = int(intersect)
state = env.encode_state(intersect,intersect,intersect,intersect)
env.state = state
if env.start == env.state:
  print("we are in start position state:", env.state)
else:
  print("state:", env.state)
transitions_val.append(env.R[env.state])
#print("destination:",env.goal)
#print(env.R[36])
img = env.render()
axs[2].imshow(img)
axs[2].set_title("State with traffic true")
env.trafic_ON = False

#we set a state where we are chrashed in a building only acceptable action is getting out of the wall (West in this case)
#print("\n setting car crash, nooo")
print("\n setting state to state when crashed")
state = env.encode_state(2,9,0,0)
env.state = state
if env.start == env.state:
  print("we are in start position state:", env.state)
else:
  print("state:", env.state)
#print("destination:",env.goal)
#print(env.R[29])
transitions_val.append(env.R[env.state])
img = env.render()
axs[3].imshow(img)
axs[3].set_title("State when crashed")

print("\n reset environment state:",env.reset())
#print("state: ", env.state)
img = env.render()
transitions_val.append(env.R[env.state])
axs[4].imshow(img)
axs[4].set_title("Environment reset")


print("\nThese are the transition values of the mentioned states\n",np.asarray(transitions_val))