import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridEnvironment(gym.Env):
    metadata = { 'render.modes': [] }
    
    def __init__(self,env="Deterministic", max_timesteps=100):

        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Discrete(4)
        self.max_timesteps = max_timesteps
        self.env = env
        self.timestep = 0
        self.agent_pos = [0, 0]
        self.goal_pos = [4, 4]
        self.reward1_pos = [1,4]
        self.reward2_pos = [2,3]
        self.reward3_pos = [3,1]
        self.reward4_pos = [1,1]
        self.state = np.zeros((5,5))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        self.state[tuple(self.reward1_pos)] = 0.1
        self.state[tuple(self.reward2_pos)] = 0.3
        self.state[tuple(self.reward3_pos)] = 0.8
        self.state[tuple(self.reward4_pos)] = 0.7

        
    def reset(self, **kwargs):
        self.agent_pos = [0, 0]
        self.state = np.zeros((5,5))
        self.goal_pos = [4, 4]
        self.reward1_pos = [1,4]
        self.reward2_pos = [2,3]
        self.reward3_pos = [3,1]
        self.reward4_pos = [1,1]
        self.timestep=0
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        self.state[tuple(self.reward1_pos)] = 0.1
        self.state[tuple(self.reward2_pos)] = 0.3
        self.state[tuple(self.reward3_pos)] = 0.8
        self.state[tuple(self.reward4_pos)] = 0.7
        observation = self.state.flatten()

        info = {}

        return observation, info
    
    def step(self, action):
        if self.env=="Stochastic":
          l=[0,1,2,3]
          action = np.random.choice([action] + list(filter(lambda a: a!=action, l)),p=[0.4,0.2,0.2,0.2])

        if action == 0:
          self.agent_pos[0] += 1
        if action == 1:
          self.agent_pos[0] -= 1
        if action == 2:
          self.agent_pos[1] += 1
        if action == 3:
          self.agent_pos[1] -= 1

        self.agent_pos = np.clip(self.agent_pos, 0, 4)

        self.state = np.zeros((5,5))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        self.state[tuple(self.reward1_pos)] = 0.1
        self.state[tuple(self.reward2_pos)] = 0.3
        self.state[tuple(self.reward3_pos)] = 0.8
        self.state[tuple(self.reward4_pos)] = 0.7
        
        reward = 0
        if np.array_equal(self.agent_pos, self.goal_pos):
          self.state[tuple(self.goal_pos)] = 1
          reward = 10
        
        if np.array_equal(self.agent_pos, self.reward1_pos):
          self.state[tuple(self.reward1_pos)] = 1
          reward = -1


        if np.array_equal(self.agent_pos, self.reward2_pos):
          self.state[tuple(self.reward2_pos)] = 1
          reward = -0.5


        if np.array_equal(self.agent_pos, self.reward3_pos):
          self.state[tuple(self.reward3_pos)] = 1
          reward = -1.5


        if np.array_equal(self.agent_pos, self.reward4_pos):
          self.state[tuple(self.reward4_pos)] = 1
          reward = -2
        
        observation = self.state.flatten()
        
        self.timestep += 1

        terminated = True if np.array_equal(self.agent_pos,self.goal_pos) else False
        truncated = True if self.timestep >= self.max_timesteps else False
        info = {}
        return observation, reward, terminated, truncated, info
        
    def render(self):
        plt.imshow(self.state)