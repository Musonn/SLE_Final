import gym
env = gym.make('CartPole-v0')

# define hyperparameters
epoch = 500

def policy(obs):
    angle = obs[2]
    return 0 if angle<0 else 1

totals =[]
for episode in range(epoch):
    print(episode)
    episode_reward = 0
    obs = env.reset()
    for step in range(1000):
        env.render()
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            break
    totals.append(episode_reward)
env.close()
