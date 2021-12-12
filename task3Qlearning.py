from typing import Deque
import gym
import numpy as np
import random
import math
env = gym.make('CartPole-v0')

# Set random generator for reproductible runs
env.seed(0)
np.random.seed(0)

# initialization
all_actions = env.action_space.n    # int 2
q_value_table = np.zeros((3,3,6,6) + (all_actions,)) # the magic number refers to the page 5 of instruction
x_history = list(); angle_history = list()


# user-defined parameters
min_explore_rate = 0.001; min_learning_rate = 0.1; max_episodes = 1000
discount = 0.99
score = Deque(maxlen = 1000)

def get_state(observation):
    a,b,c,d = observation   # x, x', theta, theta'
    c=c/np.pi*180
    d=d/np.pi*180
    if a < -1.6:
        a = 1
    elif a < 1.6:
        a = 2
    else:
        a = 3

    if b < -0.5:
        b=1
    elif b < 0.5:
        b=2
    else:
        b = 3
    
    if c < -16:
        c=1
    elif c < -8:
        c=2
    elif c < 0:
        c=3
    elif c < 8:
        c=4
    elif c < 16:
        c=5
    else:
        c=6
    
    if d < -50:
        d=1
    elif d < -30:
        d=2
    elif d < -10:
        d=3
    elif d < 10:
        d=4
    elif d < 30:
        d=5
    else:
        d=6
    return tuple([a-1, b-1, c-1, d-1])

def select_learning_rate(x):
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((x+1)/70)))

# change the exploration rate over time.
def select_explore_rate(x):
    return max(min_explore_rate, min(1.0, 1.0 - math.log10((x+1)/70)))

def select_action(state_value, explore_rate): 
    if random.random() < explore_rate:
        action = env.action_space.sample() # explore
    else: 
        action = np.argmax(q_value_table[state_value])# exploit
    return action

# train the system
totaltime = 0
time_step_history = 0
for episode_no in range(max_episodes):
    # initialize the environment
    obs = env.reset()
    prev_state = get_state(obs)
    learning_rate = select_learning_rate(episode_no)
    explore_rate = select_explore_rate(episode_no)
    done = False
    time_step = 0   # use time_step to represent reward
    while not done:
        #env.render()
        # select action using epsilon-greedy policy
        action = select_action(prev_state, explore_rate)
        # record new observations
        obs, reward_gain, done, info = env.step(action)
        if done: break
        #update q_value_table
        best_q = max(q_value_table[get_state(obs)])
        sample = reward_gain + discount * best_q
        q_value_table[prev_state][action] += learning_rate * (sample - q_value_table[prev_state][action])
        # update the states for next iteration
        prev_state = get_state(obs)

        time_step += 1
        # while loop ends here
        if episode_no == 999:
            x_history.append(obs[0])
            angle_history.append(obs[2])
    score.append(time_step)
env.close()

# output the results
# f = open('result_Q.txt','a')    # remember to change the working directory in your terminal / console
# for number in score:        # the default is C:\Users\whatever_user_name_it_is
#     f.write(str(number))
#     f.write(' ')
# f.write('\n')
# f.close()

# with open('position and angle.txt', 'w') as g:
#     for i in x_history:
#         g.write(str(i)+' ')
#     g.write('\n')
#     for i in angle_history:
#         g.write(str(i)+' ')
#     g.write('\n')