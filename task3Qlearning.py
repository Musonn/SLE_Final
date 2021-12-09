import gym
import numpy as np
import random
env = gym.make('CartPole-v0')

# Set random generator for reproductible runs
env.seed(0)
np.random.seed(0)

# define hyperparameters
epoch = 500
explore_rate = 0.5
learning_rate = 0.5
# user-defined parameters
min_explore_rate = 0.1; min_learning_rate = 0.1; max_episodes = 1000
max_time_steps = 250; streak_to_end = 120; solved_time = 199; discount = 0.99
no_streaks = 0

# initialization
start_state = (3,2,2,2)
all_actions = env.action_space.n    # int 2
q_value_table = np.zeros((6,3,3,3) + (all_actions,)) # the magic number refers to the page 5 of instruction

# user-defined parameters
min_explore_rate = 0.1; min_learning_rate = 0.1; max_episodes = 1000
max_time_steps = 250; streak_to_end = 120; solved_time = 199; discount = 0.9
no_streaks = 0


def policy(obs):    # obs is a list of theta, x, theta', x'
    angle = obs[2]
    return 0 if angle<0 else 1

def get_state(observation):
    a, b, c, d = observation
    if a < -6:
        a = 1
    elif a < -1:
        a =2
    elif a < 0:
        a =3
    elif a < 1:
        a = 4
    elif a < 6:
        a=5
    else:
        a=6
    
    if b < -0.8:
        b=1
    elif b < 0.8:
        b=2
    else:
        b = 3

    if c < -50:
        c=1
    elif c < 50:
        c=2
    else:
        c = 3
    
    if d < -0.5:
        d=1
    elif d < 0.5:
        d=2
    else:
        d = 3

    return  tuple([a-1,b-1,c-1,d-1])


def select_action(state_value, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample() # explore
    else: # exploit
        action = np.argmax(q_value_table[state_value])
    return action

# train the system
totaltime = 0
time_step_history = 0
for episode_no in range(max_episodes):
    # initialize the environment
    prev_state = start_state
    obs = env.reset()
    done = False
    time_step = 0
    while not done:
        #env.render()
        # select action using epsilon-greedy policy
        action = select_action(prev_state, explore_rate)
        # record new observations
        obs, reward_gain, done, info = env.step(action)
        #update q_value_table
        best_q = max(q_value_table[prev_state])
        q_value_table[prev_state][action] = (1-learning_rate) * q_value_table[prev_state][action] + \
        learning_rate * (reward_gain + discount * best_q)
        # update the states for next iteration
        prev_state = get_state(obs)
        time_step += 1
        # while loop ends here

env.close()