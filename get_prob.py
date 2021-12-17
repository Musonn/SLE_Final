import gym
import numpy as np
from numpy.lib.function_base import angle
from random import seed, uniform
from math import pi, sin, cos

seed(0)

shape = (5,3,6,6)
all_actions = 2

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

        # check if to stop
        self.steps_beyond_done = None

    def step(self, state_parameters, action):

        x, x_dot, theta, theta_dot = state_parameters
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = cos(theta)
        sintheta = sin(theta)
        sgn = lambda x : (x>0) - (x<0) # https://thingspython.wordpress.com/2011/03/12/snippet-sgn-function/

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        temp2 = self.mu_p*theta_dot/self.polemass_length
        thetaacc = (self.gravity * sintheta - costheta * (temp - self.mu_c*sgn(x_dot)/self.total_mass) - temp2) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass - self.mu_c*sgn(x_dot) / self.total_mass

        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        # done = bool(
        #     x < -self.x_threshold
        #     or x > self.x_threshold
        #     or theta < -self.theta_threshold_radians
        #     or theta > self.theta_threshold_radians
        # )

        # if not done:
        #     reward = 0
        # elif self.steps_beyond_done is None:
        #     # Pole just fell!
        #     self.steps_beyond_done = 0
        #     reward = -1
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1
        #     reward = 0

        return np.array(self.state, dtype=np.float32)

def get_p(state, action):
    '''
    This function compute a table of transition probability based on current state
    and the action. Huge time complexity.
    '''
    iterations = int(1e3)
    model = DPEnv()
    p_table = np.zeros((5,3,6,6,2))
    for i in range(iterations):
        state_parameters = get_parameters(state)
        next_state = model.step(state_parameters, action)
        p_table[get_state(next_state)][action]+=1

    p_table = p_table/iterations

    return p_table

def get_parameters(state):
    '''
    Ramdomly choose an observation from a state
    '''
    a, b, c, d = state
    
    if a == 0:
        x_range = (-4.8,-2.4)
    elif a ==1:
        x_range = (-2.4,-0.8)
    elif a ==2:
        x_range = (-0.8,0.8)
    elif a ==3:
        x_range = (0.8,2.4)
    elif a ==4:
        x_range = (2.4,4.8)

    if b == 0:
        x_v_range = (-50,-0.5)
    elif b ==1:
        x_v_range = (-0.5,0.5)
    elif b ==2:
        x_v_range = (0.5,50)

    if c == 0:
        angle_range = (-24,-16)
    elif c ==1:
        angle_range = (-16,-8)
    elif c == 2:
        angle_range = (-8,0)
    elif c == 3:
        angle_range = (0,8)
    elif c == 4:
        angle_range = (8,16)
    elif c == 5:
        angle_range = (16,24)

    if d == 0:
        angle_v_range = (-100,-50)  # tehnically min is -inf. Just pick -100 for no reason
    elif d == 1:
        angle_v_range = (-50,-30)
    elif d == 2:
        angle_v_range = (-30,-10)
    elif d == 3:
        angle_v_range = (-10,10)
    elif d == 4:
        angle_v_range = (10,30)
    elif d == 5:
        angle_v_range = (30,100)

    

    return uniform(x_range[0],x_range[1]), uniform(x_v_range[0], x_v_range[1]), \
        uniform(angle_range[0]*pi/180,angle_range[1]*pi/180), uniform(angle_v_range[0]*pi/180, angle_v_range[1]*pi/180)

def get_state(observation):
    '''
    Match obervation to the state
    '''
    a,b,c,d = observation   # x, x', theta, theta'
    c=c/np.pi*180
    d=d/np.pi*180
    if a < -2.4:
        a = 1
    elif a < -0.8:
        a = 2
    elif a < 0.8:
        a = 3
    elif a < 2.4:
        a=4
    else:
        a=5

    if b < -0.5:
        b=1
    elif b < 0.5:
        b=2
    else:
        b = 3
    
    if c < -12:
        c=1
    elif c < -6:
        c=2
    elif c < 0:
        c=3
    elif c < 6:
        c=4
    elif c < 12:
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

def get_reward(state):
    a,b,c,d = state
    if 0<a and a<4:
        if 0<c and c<5:
            return 0
        else:
            return -1
    else:
        return -1 
# state = (1,1,4,0)
# a = get_reward(state)
# print(a)