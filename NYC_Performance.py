# Agent's performance after Q-learning with traffic off between 2 and 7 and off after 11 
total_epochs, total_penalties = 0, 0
episodes = 1
frames = []
states_eval = []
for _ in range(episodes):
  state = env.reset()
  epochs, step, reward = 0, 0, 0
  reward_until_goal = 0
  done = False
  while not done:
    # ============================= Traffic on or off ======================
    if step > 3 and step < 7:
      env.trafic_ON = False
    elif step > 11:
      env.trafic_ON = False
    else:
      env.trafic_ON = True
    # ============================= Encode the state =======================
    x,y = env.decode_state(env.state)[0], env.decode_state(env.state)[1]
    if env.trafic_ON:
      env.state = env.encode_state(x,y,x,y)
    if env.trafic_ON is False:
      env.state = env.encode_state(x,y,0,0)
    state = env.state
    print("step", step, "env traffic", env.trafic_ON, "state", state)
    step += 1
    action = np.argmax(Q[state])
    # ======================== Keeping track of metrics ====================
    Prob, Next_state, Reward, Done = env.step(action)
    state = Next_state
    states_eval.append(state)
    env.state = Next_state
    done = Done
    epochs += 1
    reward += Reward
    frames.append({'frame': env.render(states_eval), 'state': state, 'action': action, 'reward': reward})
  reward_until_goal += (reward/epochs)
  print("reached goal")
  total_epochs += epochs

print("average reward until goal: ", reward_until_goal/episodes)
print(f"Average timesteps per episode: {total_epochs / episodes}")

imgs = print_frames(frames)


"""## Let's try parameter tuning"""

############### training #############
import time
start_time = time.time()
from IPython.display import clear_output
import random
import pandas as pd
import math

print( 20 * "#" + " Attention resetting Q matrix " + 20 * "#")
Q = np.zeros([env.observation_space.n, env.action_space.n])
print("Q matrix shape\n",Q.shape)

alpha = 0.1
gamma = 0.2

while gamma <= 1:
  ##### epsilon values ####
  epsilon = 0.7  ############
  min_epsilon = 0.05 #######
  epsilon_decay = 1e-5 ####
  #########################
  # random amount of times during training where traffic is true
  threshold_traffic = 0.8
  ########################
  epochs = []
  average_rewards = []
  steps = []
  steps_until_goal = []
  reward_until_goal_list = []
  stop = False
  print("start state: ", env.start, "destination: ", env.goal)
  start_Q = np.zeros((2,4))
  for i in range(1,50_000): #We need to train very long to make sure we train for traffic and without traffic
    state = env.reset()
    #print("state", state)
    epochs, reward = 0,0
    step_until_goal = 0
    total_reward = 0
    reward_until_goal = 0
    done = False
    steps_traffic = 0
    steps_no_traffic = 0 
    while not done:
      ###### traffic on or off ########
      if random.uniform(0,1) > threshold_traffic: #make traffic true or false
        env.trafic_ON = True
        steps_traffic += 1
      else:
        env.trafic_ON = False
        steps_no_traffic += 1
      
      x,y = env.decode_state(env.state)[0], env.decode_state(env.state)[1]
      if env.trafic_ON:
        env.state = env.encode_state(x,y,x,y)
      if env.trafic_ON is False:
        env.state = env.encode_state(x,y,0,0)
      
      ######## epsilon greedy
      state = env.state
      if random.uniform(0,1) < epsilon: 
        action = env.action_space.sample() # explore 🧐
      else:
        action = np.argmax(Q[state]) # exploit 😈

      Prob, Next_state, Reward, Done = env.step(action)
      old_value = Q[state,action]
      next_max = np.max(Q[Next_state])
      old_state = state
      new_value = (1 - alpha)*old_value + alpha*(Reward + gamma + next_max)
      Q[state, action] = new_value
      
      ### metrics ####
      total_reward += Reward
      reward_until_goal += Reward
      state = Next_state
      env.state = state
      epochs += 1
      done = Done
      step_until_goal += 1

      
      #### append metrics to list ###
      if done:
        steps_until_goal.append(step_until_goal)
        reward_until_goal_list.append(reward_until_goal)

  