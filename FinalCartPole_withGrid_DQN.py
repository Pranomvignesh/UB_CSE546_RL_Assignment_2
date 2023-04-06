import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from dqn import DqnOptions,DQN,DuelingDQN,Hyperparams
from gridEnvironment import GridEnvironment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetworkForCartPole(nn.Module):
    def __init__(self, observation_space, action_space, learningRate):
        super().__init__()
        self.layer_1 = nn.Linear(observation_space, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, action_space)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

dqnCartPole = DQN(
    env=gym.make('CartPole-v1'),
    hyperparams=Hyperparams(
        epsilon=0.99,
        epsilonMin=0.001,
        epsilonDecay=10000,
        memorySize=10000,
        learningRate=1e-3,
        batchSize=128,
        discountFactor=0.99,
        targetNetworkUpdateFrequency=20,
        episodes=2000
    ),
    nnModel=NeuralNetworkForCartPole,
    options = DqnOptions(
        resultsPath='./results',
        filePrefix='CartPole_DQN_',
        showLiveResults=True
    )
)

dqnCartPole.train()

class NeuralNetworkForGridEnv(nn.Module):
    def __init__(self, env, learningRate):
        super().__init__()
        input_shape = env.observation_space.n
        self.layer_1 = nn.Linear(input_shape, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, env.action_space.n)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

dqnGridEnv = DQN(
    env=GridEnvironment(env="Deterministic", max_timesteps=100),
    hyperparams=Hyperparams(
        epsilon=0.9,
        epsilonMin=0.001,
        epsilonDecay=10000,
        memorySize=1000,
        learningRate=0.001,
        batchSize=64,
        discountFactor=0.99,
        targetNetworkUpdateFrequency=20,
        episodes=600
    ),
    nnModel=NeuralNetworkForGridEnv
)

dqnGridEnv.train()

class DuelingNetworkForCartPole(nn.Module):
    def __init__(self, observation_space, action_space, learningRate):
        super().__init__()
        self.layer_1 = nn.Linear(observation_space, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, action_space)
        self.layer_4 = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        advantage = self.layer_3(x)
        value = self.layer_4(x)
        q = value + (advantage - torch.mean(advantage, dim=1, keepdim=True))
        return q

duelingDqnCartPole = DuelingDQN(
    env=gym.make('CartPole-v1'),
    hyperparams=Hyperparams(
        epsilon=0.99,
        epsilonMin=0.0001,
        epsilonDecay=10000,
        memorySize=10000,
        learningRate=0.01,
        batchSize=64,
        discountFactor=0.994,
        targetNetworkUpdateFrequency=20,
        episodes=1500,
        tau = 0.005
    ),
    nnModel=DuelingNetworkForCartPole
)

duelingDqnCartPole.train()

class DuelingNetworkForGridEnv(nn.Module):
    def __init__(self, env, learningRate):
        super().__init__()
        input_shape = env.observation_space.n
        self.layer_1 = nn.Linear(input_shape, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, env.action_space.n)
        self.layer_4 = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        advantage = self.layer_3(x)
        value = self.layer_4(x)
        q = value + (advantage - torch.mean(advantage, dim=1, keepdim=True))
        return q

duelingDQNGridEnvironment = DuelingDQN(
    env=GridEnvironment(env="Deterministic", max_timesteps=100),
    hyperparams=Hyperparams(
        epsilon=0.9,
        epsilonMin=0.001,
        epsilonDecay=10000,
        memorySize=1000,
        learningRate=0.001,
        batchSize=64,
        discountFactor=0.99,
        targetNetworkUpdateFrequency=20,
        episodes=600,
        tau = 0.005
    ),
    nnModel=DuelingNetworkForGridEnv
)

duelingDQNGridEnvironment.train()