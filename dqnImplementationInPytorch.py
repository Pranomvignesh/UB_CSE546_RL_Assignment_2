import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', (
    'state',
    'action',
    'reward',
    'next_state',
    'done'
))
HyperParameters = namedtuple('HyperParameters', (
    'memory_size',
    'epsilon',
    'epsilon_min',
    'epsilon_decay',
    'learning_rate',
    'discount_factor',
    'batch_size',
    'noOfEpisodes',
    'stepsPerEpisode',
    'tau'
))


class ReplayMemory():
    def __init__(self, size) -> None:
        self.size = size
        self._memory = []
        pass

    def add(self, *args):
        if len(self._memory) >= self.size:
            self._memory.pop(0)
        self._memory.append(Transition(*args))

    def sampleMinibatch(self, batch_size):
        if len(self._memory) < batch_size:
            return None
        return random.sample(self._memory, batch_size)


class NeuralNetwork(nn.Module):

    def __init__(self, observation_space, action_space):
        super(NeuralNetwork, self).__init__()
        self.linearLayer_1 = nn.Linear(observation_space, 64)
        self.linearLayer_2 = nn.Linear(64, 128)
        self.linearLayer_3 = nn.Linear(128, 32)
        self.linearLayer_4 = nn.Linear(32, action_space)

    def forward(self, x):
        x = nn.functional.relu(self.linearLayer_1(x))
        x = nn.functional.relu(self.linearLayer_2(x))
        x = nn.functional.relu(self.linearLayer_3(x))
        return self.linearLayer_4(x)


class DQN():
    def __init__(self, env, hyperparams: HyperParameters, nnModel: NeuralNetwork) -> None:
        self.env = env
        self.action_space = self.env.action_space.n
        self.observation_space = len(self.env.observation_space)
        self.hyperparams = hyperparams
        self.nnModel = nnModel
        self.init_networks()
        self.init_memory_buffer()
        pass

    def init_networks(self):
        self.policyQNetwork = self.nnModel(
            self.observation_space, self.action_space).to(DEVICE)
        self.targetQNetwork = self.nnModel(
            self.observation_space, self.action_space).to(DEVICE)
        self.targetQNetwork.load_state_dict(self.policyQNetwork.state_dict())
        self.optimizer = optim.AdamW(
            self.policyQNetwork.parameters(),
            lr=self.hyperparams.learning_rate, amsgrad=True)

    def init_memory_buffer(self):
        self.memory_buffer = ReplayMemory(size=self.hyperparams.memory_size)

    def preprocessState(self, state):
        return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    def getAction(self, state):
        if np.random.rand() < self.hyperparams.epsilon:
            return torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policyQNetwork(state).max(1)[1].view(1, 1)

    def optimizeNetwork(self, batch):
        batch = Transition(*zip(*batch))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.done)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([next_state for next_state, done in zip(batch.next_state, batch.done)
                                           if done is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policyQNetwork(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(len(batch), device=DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.targetQNetwork(
                non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.hyperparams.discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policyQNetwork.parameters(), 100)
        self.optimizer.step()

    def updatePlot(self, episodes, show_result=False):
        fig = plt.figure(1)
        episodes = torch.tensor(episodes, dtype=torch.float)
        if show_result:
            plt.suptitle('Result')
        else:
            plt.clf()
            plt.suptitle('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(episodes.numpy())
        # Take 100 episode averages and plot them too
        if len(episodes) >= 100:
            means = episodes.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.title('Current Mean Value :' + str(means[-1].item()))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def train(self):
        env = self.env
        noOfEpisodes = self.hyperparams.noOfEpisodes
        stepsPerEpisode = self.hyperparams.stepsPerEpisode
        epsilon = self.hyperparams.epsilon
        epsilon_decay = self.hyperparams.epsilon_decay
        epsilon_min = self.hyperparams.epsilon_min
        batch_size = self.hyperparams.batch_size
        for episode in noOfEpisodes:
            state, info = env.reset()
            self.preprocessState(state)
            for step in stepsPerEpisode:
                action = self.getAction(state)
                next_state, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=DEVICE)
                next_state = self.preprocessState(next_state)
                self.memory_buffer.add(
                    state, action, reward, next_state, done
                )
                minibatch = self.memory_buffer.sampleMinibatch(batch_size)
                if minibatch:
                    self.optimizeNetwork(minibatch)

                tau = self.hyperparams.tau
                target_weights = self.targetQNetwork.state_dict()
                policy_weights = self.policyQNetwork.state_dict()
                for key in policy_weights:
                    target_weights[key] = policy_weights[key] * \
                        tau + target_weights[key]*(1-tau)
                self.targetQNetwork.load_state_dict(target_weights)

                if done:
                    self.updatePlot(episode)
                    break

            epsilon *= epsilon_decay
            self.epsilon = max(epsilon_min, epsilon)


env = gym.make("CartPole-v1")
hyperparams = HyperParameters(
    memory_size=1000,
    epsilon=1,
    epsilon_min=0.01,
    epsilon_decay=0.99,
    learning_rate=1e-4,
    discount_factor=0.99,
    batch_size=128,
    noOfEpisodes=2000,
    stepsPerEpisode=1000,
    tau=0.005
)

dqn = DQN(
    env=env,
    hyperparams=hyperparams,
    nnModel=NeuralNetwork
)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


print('Complete')
dqn.updatePlot(episodes=2000, show_result=True)
plt.ioff()
plt.show()
