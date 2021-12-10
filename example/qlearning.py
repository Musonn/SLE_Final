from environment import *
import numpy as np

# Original code from https://ferdinand-muetsch.de/cartpole-with-qlearning-first-experiences-with-openai-gym.html
import math, random
# Ferdinand Mütsch ran a GridSearch on hyperparameters:
#   Best hyperparameters: 'buckets': (1, 1, 6, 12), 'min_alpha': 0.1, 'min_epsilon': 0.1

# Define a Q-Learning Policy
class Policy_QLearning():
    def __init__(self, buckets=(1, 1, 6, 12,), min_alpha=0.1, min_epsilon=0.1, gamma=1.0, ada_divisor=25, state_space_dim=4, action_space_dim=2):
        self.buckets = buckets # down-scaling feature space to discrete range
        self.min_alpha = min_alpha # Learning Rate
        self.min_epsilon = min_epsilon # Exploration Rate: # to avoid local-minima, instead of the best action, chance ε of picking a random action
        self.gamma = gamma # Discount Rate factor
        self.ada_divisor = ada_divisor # only for development purposes

        self.action_space_dim = action_space_dim
        self.Q = np.zeros(self.buckets + (self.action_space_dim,))
        self.rewards = [0]  # Initialize to 0
        self.action = 0  # Initialize to 0

    def discretize(self, obs):
        # Ferdinand choose to
        upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
        lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def act(self, state):
        new_state = self.discretize(state)
        self.update_q(self.current_state, self.action, self.rewards[-1], new_state, self.alpha)
        self.current_state = new_state

        if np.random.rand() <= self.epsilon:
            self.action = random.randrange(self.action_space_dim)
            return self.action
        self.action = np.argmax(self.Q[self.current_state])
        return self.action

    def memorize(self, next_state, action, reward, done):
        self.rewards.append(reward)

    def update_q(self, state_old, action, reward, state_new, alpha):
        best_q = max(self.Q[state_new])
        sample = reward + 1 * best_q
        self.Q[state_old][action] += self.alpha * (reward + 1 * best_q - self.Q[state_old][action])
    def reset(self, state):
        # Decrease alpha and epsilon while experimenting
        t = len(self.rewards)
        self.alpha = max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))
        self.epsilon = max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))
        self.current_state = self.discretize(state)

policy = Policy_QLearning()
scores, trials_to_solve = run_cartPole(policy, n_episodes=2000, print_every=100, render_env=False)
print('** Mean average score:', np.mean(scores))
plot_performance(scores, policy)
plot_trials_to_solve(trials_to_solve)