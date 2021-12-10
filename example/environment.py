# Run the environment

from collections import deque
import numpy as np
from gym.envs.classic_control.cartpole import *
from pyglet.window import key
import time

bool_quit = False

# Define the Environments
env = gym.make('CartPole-v0')

# We can customize the environments:
# env.env.theta_threshold_radians = np.pi * 2
# env.env.x_threshold = 5
# env.env.force_mag = 100
# env.env.state[2] = np.pi  # Initial angle state to put after env.reset()

# Set random generator for reproductible runs
env.seed(0)
np.random.seed(0)

def key_press(k, mod):
    global bool_quit, action, restart
    if k==0xff0d: restart = True
    if k==key.ESCAPE: bool_quit=True  # Quit properly
    if k==key.Q: bool_quit=True  # Quit properly
    if k==key.LEFT:  action = 0  # 0        Push cart to the left
    if k==key.RIGHT: action = 1  # 1        Push cart to the right

def run_cartPole(policy, n_episodes=1000, max_t=1000, print_every=100, render_env=True, record_video=False):
    """Run the CartPole-v0 environment.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): how often to print average score (over last 100 episodes)

    Adapted from:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/hill-climbing/Hill_Climbing.ipynb

    """
    global bool_quit

    if policy.__class__.__name__ == 'Policy_Human':
        global action
        action = 0  # Global variable used for manual control with key_press
        env = CartPoleEnv()  # This is mandatory for keyboard input
        env.reset()  # This is mandatory for keyboard input
        env.render()  # This is mandatory for keyboard input
        env.viewer.window.on_key_press = key_press  # Quit properly & human keyboard inputs
    else:
        print('** Evaluating', policy.__class__.__name__, '**')
        # Define the Environments
        env = gym.make('CartPole-v0')
        # Set random generator for reproductible runs
        env.seed(0)
        np.random.seed(0)

    if record_video:
        env.monitor.start('/tmp/video-test', force=True)

    scores_deque = deque(maxlen=100)
    scores = []
    trials_to_solve=[]

    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = env.reset()
        if 'reset' in dir(policy):  # Check if the .reset method exists
            policy.reset(state)
        for t in range(max_t):  # Avoid stucked episodes
            action = policy.act(state)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            if 'memorize' in dir(policy):  # Check if the .memorize method exists
                policy.memorize(state, action, reward, done)
            if render_env: # Faster, but you can as well call env.render() every time to play full window.
                env.render()  # (mode='rgb_array')  # Added  # Changed mode

            if done:  # if Pole Angle is more than +-12 deg or Cart Position is more than +-2.4 (center of the cart reaches the edge of the display) the simulation ends
                trials_to_solve.append(t)
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        if 'update' in dir(policy):  # Check if the .update method exists
            policy.update(state)  # Update the policy

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}\tSteps: {:d}'.format(i_episode, np.mean(scores_deque), t))
        if np.mean(scores_deque) >= 195.0:
            print('Episode {}\tAverage Score: {:.2f}\tSteps: {:d}'.format(i_episode, np.mean(scores_deque), t))
            print('** Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(max(1, i_episode-100), np.mean(scores_deque)))
            break
        if bool_quit:
            break

    if np.mean(scores_deque) < 195.0:
        print('** The environment has never been solved!')
        print('   Mean scores on all runs was < 195.0')
    if record_video:
        env.env.close()
    env.close()
    return scores, trials_to_solve

#  Performance plots
import matplotlib.pyplot as plt

def plot_performance(scores, policy):
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
    plt.title(policy.__class__.__name__ +  ' performance on CartPole-v0')
    plt.show()

def plot_trials_to_solve(trials_to_solve, policy):
    # Plot the policy number of trials to solve the Environment
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(trials_to_solve, bins='auto', density=True, facecolor='g', alpha=0.75)
    plt.ylabel('Frequency')
    plt.xlabel('Number of Trial to solve')
    plt.title(policy.__class__.__name__ +  ' trials to solve CartPole-v0')
    plt.show()

def plot_weights_history(weights_history):
    weights_history = np.array(weights_history)
    # weights_history[:, 0, 0]  # Cart Pos, Action 0
    # weights_history[:, 1, 0]  # Cart Vel, Action 0
    # weights_history[:, 2, 0]  # Pole Ang, Action 0
    # weights_history[:, 3, 0]  # Pole Vel, Action 0
    plt.figure(0)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=1.6, hspace=.5)
    i = 0
    for obs_id in range(weights_history.shape[1]):  # Loop through Observations
        if len(weights_history.shape) == 3:
            for act_id in range(weights_history.shape[2]):  # Action 0 and Action 1
                plt.subplot(4, 2, i + 1)
                i += 1
                # plt.axis('off')
                plt.plot(weights_history[:, obs_id, act_id])
                plt.title('Obs%d, Act%d' % (obs_id, act_id))
        else:
            plt.subplot(2, 2, i + 1)
            i += 1
            # plt.axis('off')
            plt.plot(weights_history[:, obs_id])
            plt.title('Obs%d' % (obs_id))