from copy import deepcopy
import numpy as np
import random
import cv2
import gym

class Env():
    def __init__(self):
        self.width, self.height = 256, 256
        self.length = 100

        #physics
        self.x = 0
        self.y = 0
        self.theta = 0
        self.a = -0.1
        self.b = 0.13
        self.r = 0.05
        self.time_step = 0.01
        self.dt = 0.001

        #hazard
        self.hazard_list = np.array([[1.5, 0.0], [-1.5, 0.0], [0.0, 1.5], [0.0, -1.5],
                                    [3.0, 3.0], [-3.0, 3.0], [3.0, -3.0], [-3.0, -3.0]])
        self.hazard_radius = 1.0

        self.steps = 0
        self.max_steps = 500

        state_dim = 3
        action_dim = 2
        self.observation_space = gym.spaces.Box(-np.inf*np.ones(state_dim), np.inf*np.ones(state_dim), dtype=np.float32)
        self.action_space = gym.spaces.Box(np.zeros(action_dim), 2*np.ones(action_dim), dtype=np.float32)

    def reset(self):
        self.x = np.random.uniform(-5.0, 5.0)
        self.y = np.random.uniform(-5.0, 5.0)
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.steps = 0
        cv2.destroyAllWindows()
        return self.get_state()

    def step(self, action):
        assert len(action) == 2
        self.steps += 1
        u1, u2 = action
        for i in range(int(self.time_step/self.dt)):
            omega = self.r*(u2 - u1)/(2*self.b)
            vel_x = self.r*u1*np.cos(self.theta) + omega*(self.a*np.sin(self.theta) + self.b*np.cos(self.theta))
            vel_y = self.r*u1*np.sin(self.theta) + omega*(self.b*np.sin(self.theta) - self.a*np.cos(self.theta))
            self.x += vel_x*self.dt
            self.y += vel_y*self.dt
            self.theta += self.dt*omega

        next_state = self.get_state()
        reward = 0
        if self.steps < self.max_steps:
            done = False 
        else : 
            done = True
        info = {}
        return next_state, reward, done, info

    def get_state(self):
        return deepcopy([self.x, self.y, self.theta])

    def render(self):
        theta  = self.theta
        x, y = self.x, self.y
        length = 0.2
        x_range = 5.0
        width = self.width
        height = self.height

        get_y = lambda a: int((1.0 - a/x_range)*0.5*height)
        get_x = lambda a: int((1.0 + a/x_range)*0.5*width)
        get_xy = lambda a: (get_x(a[0]), get_y(a[1]))

        img = 255 * np.ones((height, width,3), np.uint8)
        cv2.circle(img, (int(width/2), int(height/2)), 5, (0,255,0), -1)
        for h_idx in range(len(self.hazard_list)):
            center = get_xy(np.array(self.hazard_list[h_idx]))
            cv2.circle(img, center, int(0.5*height*self.hazard_radius/x_range), (0,0,255), -1)
        center = np.array([x,y])
        start = center + 0.5*length*np.array([np.cos(theta), np.sin(theta)])
        start = get_xy(start)
        end = center - 0.5*length*np.array([np.cos(theta), np.sin(theta)])
        end = get_xy(end)
        cv2.line(img, start, end, (255,0,0), 5)
        cv2.imshow('image',img)
        cv2.waitKey(1)
