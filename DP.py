#Julian Brown
#SLE
#Final Project
#DP

import numpy as np

vk = np.ones((4,4))

vk = vk*-1

vk[0,0] = 0
vk[-1,-1] = 0

# define hyperparameters
theta = 0.0001
gamma = 1
v = 0

pi_right2 = np.zeros((4,4))
pi_left2 = pi_right2.copy()
pi_up2 = pi_right2.copy()
pi_down2 = pi_right2.copy()


pi_right = np.full((4,4), 0.25)
pi_right[0,0] = 0
pi_right[3,3] = 0
pi_left = pi_right.copy()
pi_up = pi_right.copy()
pi_down = pi_right.copy()

initial_pi_right = pi_right.copy()

policy_stable = False

while policy_stable == False:
    delt = 1
    while delt >= theta:
        delt = 0
        
        for i in range(4):
            for j in range(4):
                if not(i == 0 and j == 0) and not(i == 3 and j == 3):
                    
                    
                    v = vk[i,j]
                    
                    p = np.zeros((4,4))
                    
                    best_move = max(pi_right[i,j], pi_left[i,j], pi_up[i,j], pi_down[i,j])
                    count = 0
                    
                    #Right
                    if pi_right[i,j] == best_move:
                        if j == 3:
                            p[i,j] += 1
                        else:
                            p[i,j+1] += 1
                        count += 1
                    #Left
                    if pi_left[i,j] == best_move:
                        if j == 0:
                            p[i,j] += 1
                        else:
                            p[i,j-1] += 1
                        count += 1
                    #Up
                    if pi_up[i,j] == best_move:
                        if i == 0:
                            p[i,j] += 1
                        else:
                            p[i-1,j] += 1
                        count += 1
                    #Down
                    if pi_down[i,j] == best_move:
                        if i == 3:
                            p[i,j] += 1
                        else:
                            p[i+1,j] += 1
                        count += 1
                    
                    p = p/count
                    
                    summ = 0
                    for si in range(4):
                        for sj in range(4):
                            summ += p[si,sj]*(-1 + gamma*vk[si,sj])
                            
                    
                    vk[i,j] = summ
                    
                    delt = max(delt, abs(v - vk[i,j]))
                    
    for i in range(4):
        for j in range(4):
                  
            #Right
            if j == 3:
                pi_right2[i,j] = 0
            else:
                pi_right2[i,j] = vk[i,j+1] - vk[i,j]
                       
            #Left
            if j == 0:
                pi_left2[i,j] = 0
            else:
                pi_left2[i,j] = vk[i,j-1] - vk[i,j]
                
            #Up
            if i == 0:
                pi_up2[i,j] = 0
            else:
                pi_up2[i,j] = vk[i-1,j] - vk[i,j]
                
            #Down
            if i == 3:
                pi_down2[i,j] = 0
            else:
                pi_down2[i,j] = vk[i+1,j] - vk[i,j]
    
    policy_stable = True
    for i in range(4):
        for j in range(4):
            if not(i == 0 and j == 0) and not(i == 3 and j == 3):
                
                best_action = max(pi_right[i,j], pi_left[i,j], pi_up[i,j], pi_down[i,j])
                if best_action == pi_right[i,j]:
                    old_action = 'right'
                elif best_action == pi_left[i,j]:
                    old_action = 'left'
                elif best_action == pi_up[i,j]:
                    old_action = 'up'
                elif best_action == pi_down[i,j]:
                    old_action = 'down'
                    
                best_action = max(pi_right2[i,j], pi_left2[i,j], pi_up2[i,j], pi_down2[i,j])
                if best_action == pi_right2[i,j]:
                    new_action = 'right'
                elif best_action == pi_left2[i,j]:
                    new_action = 'left'
                elif best_action == pi_up2[i,j]:
                    new_action = 'up'
                elif best_action == pi_down2[i,j]:
                    new_action = 'down'
                    
                if old_action != new_action:
                    policy_stable = False
                    # print(i, '\t', j, '\t', old_action, '\t', new_action)
    
    pi_right = pi_right2.copy()
    pi_left = pi_left2.copy()
    pi_up = pi_up2.copy()
    pi_down = pi_down2.copy()

print("Final Value Function:")    
print(vk)
print("Optimal Policy (right, left, up, down): ")
#print(pi_right, '\n', pi_left, '\n', pi_up, '\n', pi_down)

optimal_policy=np.array(range(16), dtype=str).reshape(4,4)    
optimal_policy[0,0] = 'O'
optimal_policy[3,3] = 'O'
for i in range(4):
        for j in range(4):
            if not(i == 0 and j == 0) and not(i == 3 and j == 3):
                optimal_policy[i,j] = np.argmax([pi_right[i,j], pi_left[i,j], pi_up[i,j], pi_down[i,j]])
                if optimal_policy[i,j] == '0' :
                    optimal_policy[i,j] = '→'
                elif optimal_policy[i,j] == '1':
                    optimal_policy[i,j] = '←'
                elif optimal_policy[i,j] == '2':
                    optimal_policy[i,j] = '↑'
                elif optimal_policy[i,j] == '3':
                    optimal_policy[i,j] = '↓'
print(optimal_policy)