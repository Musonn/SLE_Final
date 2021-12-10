import gym
import numpy as np
import random
import math
env = gym.make('CartPole-v0')

# Set random generator for reproductible runs
env.seed(0)
np.random.seed(0)

# define hyperparameters
learning_rate = 0.1
score = []

# initialization
start_state = (3,2,2,2)
all_actions = env.action_space.n    # int 2
q_value_table = np.zeros((6,3,3,3) + (all_actions,)) # the magic number refers to the page 5 of instruction

# user-defined parameters
min_explore_rate = 0.1; min_learning_rate = 0.1; max_episodes = 2000
max_time_steps = 250; streak_to_end = 120; solved_time = 199; discount = 0.99
no_streaks = 0


def policy(obs):    # obs is a list of theta, x, theta', x'
    angle = obs[2]
    return 0 if angle<0 else 1

def get_state(observation):
    b, d, a, c = observation
    a=a/np.pi*180
    c=c/np.pi*180
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

def select_learning_rate(x):
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((x+1)/25)))

# change the exploration rate over time.
def select_explore_rate(x):
    return max(min_explore_rate, min(1.0, 1.0 - math.log10((x+1)/25)))

def select_action(state_value, episode_no):
    explore_rate = select_explore_rate(episode_no)
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
    obs = env.reset()
    prev_state = get_state(obs)
    done = False
    time_step = 0
    reward=[]
    while not done:
        #env.render()
        # select action using epsilon-greedy policy
        action = select_action(prev_state, episode_no)
        # record new observations
        obs, reward_gain, done, info = env.step(action)
        if done: reward_gain = 0
        #update q_value_table
        best_q = max(q_value_table[get_state(obs)])
        sample = reward_gain + discount * best_q
        learning_rate = select_learning_rate(episode_no)
        q_value_table[prev_state][action] = (1-learning_rate) * q_value_table[prev_state][action] + learning_rate * sample
        # update the states for next iteration
        prev_state = get_state(obs)

        reward.append(reward_gain)
        time_step += 1
        # while loop ends here
    score.append(time_step)
env.close()


#  Performance plots
import matplotlib.pyplot as plt
def plot_performance(scores):
    # Plot the policy performance
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, len(scores) + 1)
    y = scores
    plt.scatter(x, y, marker='x', c=y)
    fit = np.polyfit(x, y, deg=4)
    p = np.poly1d(fit)
    plt.plot(x,p(x),"r--")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

plot_performance(score)