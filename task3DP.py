import gym
import numpy as np
env = gym.make('CartPole-v0')

# define hyperparameters
epoch = 500

def policy(obs):
    angle = obs[2]
    return 0 if angle<0 else 1

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

# physical model for DP task
class DPEnv():
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.mu_c = 0.0005
        self.mu_p = 0.000002
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * np.pi / 180
        self.x_threshold = 2.4

    def step(self, action):

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

# hyper-parameters
threshold = 0.05 # indecisive
shape = (3,3,6,6)
all_actions = 2
discount = 0.99

# init
V_value_table = np.zeros(shape)
p = np.full(shape+(all_actions,), 0.5)
#p = np.random.normal(0, 1, shape+(all_actions,))
policy = np.full(shape, 1)  # a table full of 0 or 1
pi_right = np.zeros(shape)
pi_left = pi_right.copy()



model = DPEnv()

# policy iteration
global Evaluation
def Evaluation():
    delta = 0
    while delta < threshold:
        delta = 0
        obs = env.reset()
        for a in range(3):
            for b in range(3):
                for c in range(6):
                    for d in range(6):
                        v = V_value_table[a,b,c,d]
                        
                        summ = 0
                        for sa in range(3):
                            for sb in range(3):
                                for sc in range(6):
                                    for sd in range(6):
                                        best_move = policy[a,b,c,d]
                                        summ += p[sa,sb,sc,sd, best_move]*(1 + discount*V_value_table[sa,sb,sc,sd])
                        V_value_table[a,b,c,d] = summ
                        
                        delta = max(delta, abs(v - V_value_table[a,b,c,d]))

# Policy Improvement
policy_stable = False
eposoid = 0

while policy_stable is False:
    difference_count = 0
    eposoid+=1
    print(eposoid)
    policy_stable = True
    for a in range(3):
        for b in range(3):
            for c in range(6):
                for d in range(6):
                    old_action = policy[a,b,c,d]
                    for sa in range(3):
                        for sb in range(3):
                            for sc in range(6):
                                for sd in range(6):
                                    pi_right = np.zeros(shape)
                                    pi_left = pi_right.copy()
                                    pi_right[a,b,c,d] = p[sa,sb,sc,sd, 1]*(1 + discount*V_value_table[sa,sb,sc,sd]) # 1 is right
                                    pi_left[a,b,c,d] = p[sa,sb,sc,sd, 0]*(1 + discount*V_value_table[sa,sb,sc,sd])  # 0 is left
                                    policy[a,b,c,d] = np.argmax([pi_right[sa,sb,sc,sd], pi_left[sa, sb,sc,sd]])
                                    if old_action != policy[a,b,c,d]: policy_stable = False; difference_count+=1
    print(difference_count)
    if policy_stable is True:
        break   # the optimal policy is obtained. Break and return.
    else:
        Evaluation()

print(policy, V_value_table)
print(eposoid)