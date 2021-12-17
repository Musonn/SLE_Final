import numpy as np
from get_prob import get_state, get_p, get_reward

# hyper-parameters
threshold = 0.05 # indecisive
shape = (5,3,6,6)
all_actions = 2
discount = 0.9

# init
V_value_table = np.zeros(shape)
# p = np.full(shape+(all_actions,), 0.5)
# p = np.random.normal(0, 1, shape+(all_actions,))
policy = np.full(shape, 0)  # a table full of 0 or 1
pi_right = np.zeros(shape)
pi_left = pi_right.copy()

# value iteration
delta = threshold+1 # make sure delta > threshold at the beginning
while delta > threshold:
    delta = 0
    # iterate all states
    for a in range(1,4,1):
        for b in range(3):
            for c in range(1,5,1):
                for d in range(6):
                    v = V_value_table[a,b,c,d]
                    summ = 0
                    best_move = policy[a,b,c,d]
                    p = get_p((a,b,c,d), best_move)

                    # iterate all next_states
                    for sa in range(5):
                        for sb in range(3):
                            for sc in range(6):
                                for sd in range(6):
                                    if p[sa,sb,sc,sd, best_move] != 0:
                                        summ += p[sa,sb,sc,sd, best_move]*(get_reward((sa,sb,sc,sd)) + discount*V_value_table[sa,sb,sc,sd])
                    V_value_table[a,b,c,d] = summ
                    
                    delta = max(delta, abs(v - V_value_table[a,b,c,d]))

# setting policy by iterating all states
for a in range(1,4,1):
    for b in range(3):
        for c in range(1,5,1):
            for d in range(6):
                p_r = get_p((a,b,c,d), 1)
                p_l = get_p((a,b,c,d), 0)
                pi_right = np.zeros(shape)
                pi_left = pi_right.copy()

                # iterate all next_states
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

a=1
print('policy and value function:')
print(policy, V_value_table)