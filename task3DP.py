import gym
import numpy as np
from get_prob import get_state, get_p, get_reward
env = gym.make('CartPole-v0')

# define hyperparameters
epoch = 500

def policy(obs):
    angle = obs[2]
    return 0 if angle<0 else 1




# hyper-parameters
threshold = 0.05 # indecisive
shape = (5,3,6,6)
all_actions = 2
discount = 0.9

# init
V_value_table = np.zeros(shape)
p = np.full(shape+(all_actions,), 0.5)
#p = np.random.normal(0, 1, shape+(all_actions,))
policy = np.full(shape, 0)  # a table full of 0 or 1
pi_right = np.zeros(shape)
pi_left = pi_right.copy()

# policy iteration
global Evaluation
def Evaluation():
    delta =1e6
    while delta > threshold:
        delta = 0
        obs = env.reset()
        for a in range(1,4,1):
            for b in range(3):
                for c in range(1,5,1):
                    for d in range(6):
                        v = V_value_table[a,b,c,d]
                        
                        summ = 0
                        best_move = policy[a,b,c,d]
                        p = get_p((a,b,c,d), best_move)
                        for sa in range(5):
                            for sb in range(3):
                                for sc in range(6):
                                    for sd in range(6):
                                        if p[sa,sb,sc,sd, best_move] != 0:
                                            summ += p[sa,sb,sc,sd, best_move]*(get_reward((sa,sb,sc,sd)) + discount*V_value_table[sa,sb,sc,sd])
                        V_value_table[a,b,c,d] = summ
                        
                        delta = max(delta, abs(v - V_value_table[a,b,c,d]))
    return

# Policy Improvement
policy_stable = False
eposoid = 0

while policy_stable is False:
    difference_count = 0
    eposoid+=1
    print(eposoid)
    policy_stable = True
    for a in range(1,4,1):
        for b in range(3):
            for c in range(1,5,1):
                for d in range(6):
                    old_action = policy[a,b,c,d]
                    p_r = get_p((a,b,c,d), 1)
                    p_l = get_p((a,b,c,d), 0)
                    pi_right = np.zeros(shape)
                    pi_left = pi_right.copy()
                    for sa in range(5):
                        for sb in range(3):
                            for sc in range(6):
                                for sd in range(6):
                                    if p_r[sa,sb,sc,sd, 1] != 0:
                                        pi_right[a,b,c,d] += p_r[sa,sb,sc,sd, 1]*(get_reward((sa,sb,sc,sd)) + discount*V_value_table[sa,sb,sc,sd]) # 1 is right
                                    #else: pi_right[a,b,c,d] =0
                                    if p_l[sa,sb,sc,sd, 0] != 0:
                                        pi_left[a,b,c,d] += p_l[sa,sb,sc,sd, 0]*(get_reward((sa,sb,sc,sd)) + discount*V_value_table[sa,sb,sc,sd])  # 0 is left
                                    #else: pi_left[a,b,c,d] = 0
                    policy[a,b,c,d] = np.argmax([pi_right[a,b,c,d], pi_left[a, b,c,d]])
                    if old_action != policy[a,b,c,d]: policy_stable = False; difference_count+=1
    print(difference_count)
    if policy_stable is True:
        break   # the optimal policy is obtained. Break and return.
    else:
        Evaluation()

a=1
print(policy, V_value_table)
print(eposoid)