# ================================== TRAINING ==================================
import time
start_time = time.time()
from IPython.display import clear_output
import random
import pandas as pd

print( 20 * "#" + " Attention resetting Q matrix " + 20 * "#")
Q = np.zeros([env.observation_space.n, env.action_space.n])
print("Q matrix shape\n",Q.shape)
# ============================= Training parameters ============================
alpha = 0.1
gamma = 0.6
epsilon = 1  
min_epsilon = 0.05 
epsilon_decay = 5e-5 
# random amount of times during training where traffic is true
threshold_traffic = 0.8

# ============================= start of Training ==============================
epochs = []
average_rewards = []
steps = []
steps_until_goal = []
reward_until_goal_list = []
stop = False
print("start state: ", env.start, "destination: ", env.goal)
start_Q = np.zeros((2,4))
# we need to train for a long time to ensure the agent 
# learns with and without traffic
for i in range(1,50_000): 
  state = env.reset()
  epochs, reward = 0,0
  step_until_goal = 0
  total_reward = 0
  reward_until_goal = 0
  done = False
  steps_traffic = 0
  steps_no_traffic = 0 
  while not done:
    # ============================= Traffic on or off ==========================
    if random.uniform(0,1) > threshold_traffic: #make traffic true or false
      env.trafic_ON = True
      steps_traffic += 1
    else:
      env.trafic_ON = False
      steps_no_traffic += 1
    # ============================= Encode the state ==========================
    x,y = env.decode_state(env.state)[0], env.decode_state(env.state)[1]
    if env.trafic_ON:
      env.state = env.encode_state(x,y,x,y)
    if env.trafic_ON is False:
      env.state = env.encode_state(x,y,0,0)
    
    # ============================== Epsilon greedy ============================
    state = env.state
    if random.uniform(0,1) < epsilon: 
      action = env.action_space.sample() # explore 🧐
    else:
      action = np.argmax(Q[state]) # exploit 😈

    # =============================== Take a step ==============================
    Prob, Next_state, Reward, Done = env.step(action)
    old_value = Q[state,action]
    next_max = np.max(Q[Next_state])
    old_state = state
    new_value = (1 - alpha)*old_value + alpha*(Reward + gamma + next_max)
    Q[state, action] = new_value

       # ====================== keeping track of metrics ==========================
    total_reward += Reward
    reward_until_goal += Reward
    state = Next_state
    env.state = state
    epochs += 1
    done = Done
    step_until_goal += 1
    if done:
      steps_until_goal.append(step_until_goal)
      reward_until_goal_list.append(reward_until_goal)

  
  # ================================ Epsilon decay =============================
  epsilon = max(min_epsilon, (epsilon*(1-epsilon_decay)))

  if i % 2000 == 0:
    print("Reward until destination: ", total_reward/epochs, "***steps until goal", step_until_goal)
    print("We have taken action '{}' and this resulted in state '{}' with a reward '{}'".format(action, Next_state, Reward))
    print("During this training '{}' of steps were with traffic and '{}' of steps were without traffic".format(steps_traffic, steps_no_traffic))
    average_rewards.append(total_reward/epochs)
    steps.append(step_until_goal)
  if i % 10000 == 0:
    print(20 * "#" + " We are training with Q learning  " + 20 * "#")

# ====================== saving of the metrics =================================
steps_until_goal = np.array(steps_until_goal, dtype = 'float')
reward_until_goal = np.array(reward_until_goal_list)
df = pd.DataFrame(steps_until_goal, columns = ['Steps until goal'] )
df['Reward_until_goal'] = reward_until_goal
df.to_csv("scores_during_training.csv")
print(10* "=" + "we saved the metrics" + 10* "=")
print(20 * "#" + " We Finished training with Q learning " + 20 * "#")
print(f'the final epsilon value after training is {epsilon}')

"""# Let's plot the metrics of training"""

x = np.arange(0,len(df), 1)
fig, ax = plt.subplots(1,2, figsize =(14,5))
print(df.iloc[-1])
window = 50
std_steps = df['Steps until goal'].rolling(window).std()
ax[0].plot(x, df['Steps until goal'].rolling(window).mean(), label = 'Steps until goal', color = 'red')
ax[0].fill_between(range(len(df['Steps until goal'])), df['Steps until goal'].rolling(window).mean() - std_steps, df['Steps until goal'].rolling(window).mean() + std_steps, color = 'red', alpha = 0.1)
ax[0].set_title("Steps until goal")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Steps")
ax[0].legend(loc = 'upper right')
std_reward = df['Reward_until_goal'].rolling(window).std()
ax[1].plot(x, df['Reward_until_goal'].rolling(window).mean(), label = 'Reward until goal', color = 'red')
ax[1].fill_between(range(len(df['Reward_until_goal'])), df['Reward_until_goal'].rolling(window).mean() - std_reward, df['Reward_until_goal'].rolling(window).mean() + std_reward, color = 'red', alpha = 0.1)
ax[1].set_title("Reward until goal")
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Reward")
ax[1].legend(loc = 'lower right')
plt.show()

"""## Let's evaluate the smart agent"""

# === Agent's performance after Q-learning with traffic off between 2 and 7====
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
    if step > 2 and step < 7:
      env.trafic_ON = False
    elif step > 12:
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

def create_animation(frames):
    fig = plt.figure()
    plt.axis("off")
    im = plt.imshow(frames[0], animated=True)

    def updatefig(i):
        im.set_array(frames[i])
        return im,
    ani = animation.FuncAnimation(fig, updatefig, frames=len(frames), interval=150, blit=True)
    plt.close()
    return ani

from IPython.display import HTML
def print_frames(frames):
  imgs = []
  steps = 0
  for i, frame in enumerate(frames):
    imgs.append(frame['frame'])
    steps += 1
    if (env.decode_state(frame['state'])[0],env.decode_state(frame['state'])[1]) == (env.goal[0], env.goal[1]):
      print("we reached our goal")
      print(f"State: {frame['state']}")
      print(f"Action: {frame['action']}")
      print(f"Reward: {frame['reward']}")
      print(f"With: {steps} steps")
      plt.imshow(frame['frame'])
      plt.axis('off')
      plt.show()
      sleep(1)

  return imgs

imgs = print_frames(frames)

ani = create_animation(imgs)
HTML(ani.to_html5_video())
    