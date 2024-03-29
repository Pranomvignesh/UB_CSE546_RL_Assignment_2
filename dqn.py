import torch
import random
import gym
import pickle
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

isGymEnvLatest = int(gym.__version__.split('.')[1]) >= 26

@dataclass
class EnvInfo(object):
    env : gym.Env
    observation_space : int = None 
    action_space : int = None 

    def __post_init__(self):
        if self.observation_space == None:
            self.observation_space = self.env.observation_space.shape[0]
        if self.action_space == None:
            self.action_space = self.env.action_space.n
            

@dataclass
class Hyperparams(object):
    epsilon : float
    epsilonMin : float
    epsilonDecay : int
    learningRate : float
    batchSize : int
    discountFactor : float
    targetNetworkUpdateFrequency : int
    episodes : int
    memorySize : int
    tau : float = 0

@dataclass
class Options(object):
    resultsPath : str = ''
    filePrefix : str = ''
    showLiveResults : bool = False
    logResults : bool = True
    saveModels : bool = False
    debug : bool = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, size):
        self.memory = deque([],maxlen=size)

    def add(self, transition):
        self.memory.append(transition)

    def sample(self, batchSize):
        return random.sample(self.memory,batchSize)

    def __len__(self):
        return len(self.memory)

class DQN:
    def __init__(self, envInfo : EnvInfo, hyperparams : Hyperparams, nnModel, options : Options = None):
        self.name = type(self).__name__
        self.env = envInfo.env
        self.observation_space = envInfo.observation_space
        self.action_space = envInfo.action_space
        self.epsilon = hyperparams.epsilon
        self.epsilonMax = self.epsilon
        self.epsilonMin = hyperparams.epsilonMin
        self.epsilonDecay = hyperparams.epsilonDecay
        self.discountFactor = hyperparams.discountFactor
        self.tau = hyperparams.tau
        self.learningRate = hyperparams.learningRate
        self.updateFrequency = hyperparams.targetNetworkUpdateFrequency
        self.batchSize = hyperparams.batchSize
        self.episodes = hyperparams.episodes
        self.memory = ReplayMemory(hyperparams.memorySize)
        self.policyNetwork = nnModel(self.observation_space,self.action_space,self.learningRate)
        self.targetNetwork = nnModel(self.observation_space,self.action_space,self.learningRate)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        self.iterations = 0
        self.options = options or Options()
        self.results = ()
        resultsPath = self.options.resultsPath
        Path(f'{resultsPath}/images').mkdir(parents=True, exist_ok=True)
        Path(f'{resultsPath}/models').mkdir(parents=True, exist_ok=True)
        Path(f'{resultsPath}/weights').mkdir(parents=True, exist_ok=True)
        Path(f'{resultsPath}/rewards').mkdir(parents=True, exist_ok=True)

    def getAction(self, state, train = True):
        if train and random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(state).float().detach()
                state = state.to(device)
                state = state.unsqueeze(0)
                qValues = self.policyNetwork(state)
                return torch.argmax(qValues).item()

    def optimize(self):
        batchSize = self.batchSize
        if len(self.memory) > batchSize:
            minibatch = np.array(self.memory.sample(batchSize))
            states = minibatch[:, 0].tolist()
            actions = minibatch[:, 1].tolist()
            rewards = minibatch[:, 2].tolist()
            nextStates = minibatch[:, 3].tolist()
            dones = minibatch[:, 4].tolist()

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            nextStates = torch.tensor(
                nextStates, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)
            indices = np.arange(batchSize, dtype=np.int64)

            qValues = self.policyNetwork(states)
            qDotValues = None
            with torch.no_grad():
                qDotValues = self.targetNetwork(nextStates)

            predictedValues = qValues[indices, actions]
            predictedQDotValues = torch.max(qDotValues, dim=1)[0]

            targetValues = rewards + self.discountFactor * predictedQDotValues * dones

            loss = self.policyNetwork.loss(targetValues, predictedValues)
            self.policyNetwork.optimizer.zero_grad()
            loss.backward()
            self.policyNetwork.optimizer.step()

    def syncWeights(self):
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

    def saveResults(self):
        rewards,averageRewards, epsilons = self.results
        fig = plt.figure(1)
        fig.set_figwidth(12)
        fig.set_figheight(10)
        plt.clf()
        filePrefix = self.options.filePrefix
        title = 'Rewards vs Episodes'
        if filePrefix:
            title += ' - ' + filePrefix
        title += ' - ' + self.name
        plt.suptitle(title)
        resultsPath = self.options.resultsPath
        plt.title(f'Current Average Reward : {averageRewards[-1]}')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(rewards, '.b', label="Rewards")
        plt.plot(averageRewards, '-r', label="Average Rewards")
        plt.legend()
        if resultsPath:
            plt.savefig(f'{resultsPath}/images/{filePrefix}_{self.name}_EpisodesVsRewards.png')
        plt.clf()
        fig2 = plt.figure(1)
        fig2.set_figwidth(12)
        fig2.set_figheight(10)
        plt.title(f'Epsilon Decay - {filePrefix} {self.name}')
        plt.xlabel('Time Steps')
        plt.ylabel('Epsilon')
        
        
        plt.plot(epsilons)
        if resultsPath:
            plt.savefig(f'{resultsPath}/images/{filePrefix}_{self.name}_EpsilonDecay.png')
        plt.clf()

    def saveWeights(self):
        resultsPath = self.options.resultsPath
        filePrefix = self.options.filePrefix
        with open(f'{resultsPath}/weights/prakash8_mnunna_assignment2_{filePrefix}_{self.name}_policy_weights.pkl', 'wb') as f:
            pickle.dump(self.policyNetwork.state_dict(), f)
        with open(f'{resultsPath}/weights/prakash8_mnunna_assignment2_{filePrefix}_{self.name}_target_weights.pkl', 'wb') as f:
            pickle.dump(self.targetNetwork.state_dict(), f)

    def saveRewards(self, valueToSave):
        resultsPath = self.options.resultsPath
        filePrefix = self.options.filePrefix
        with open(f'{resultsPath}/rewards/{filePrefix}_{self.name}_rewards.pkl', 'wb') as f:
            pickle.dump(valueToSave, f)

    def saveModels(self):
        resultsPath = self.options.resultsPath
        filePrefix = self.options.filePrefix
        torch.save(self.policyNetwork,f'{resultsPath}/models/{filePrefix}_{self.name}_policy_model.pth')
        torch.save(self.targetNetwork,f'{resultsPath}/models/{filePrefix}_{self.name}_target_model.pth')

    def loadModels(self, policyModel, targetModel):
        self.policyNetwork = policyModel
        self.targetNetwork = targetModel

    def plotResults(self, rewards, averageOfLast100, epsilons = None,done=False ):
        fig = plt.figure(1)
        fig.set_figwidth(12)
        fig.set_figheight(10)
        if done:
            plt.clf()
            display.clear_output(wait=True)
            filePrefix = self.options.filePrefix
            title = 'Rewards vs Episodes'
            if filePrefix:
                title += ' - ' + filePrefix 
            title += ' - ' + self.name
            plt.suptitle(title)
            plt.title(f'Current Average Reward : {averageOfLast100[-1]}')
            plt.xlabel('Episode')
            plt.ylabel('Rewards')
            plt.plot(rewards, '.b', label="Rewards")
            plt.plot(averageOfLast100, '-r', label="Average Rewards")
            plt.legend()
            plt.show()
            if epsilons:
                fig2 = plt.figure(1)
                fig2.set_figwidth(12)
                fig2.set_figheight(10)
                plt.title(f'Epsilon Decay - {filePrefix} {self.name}')
                plt.xlabel('Time Steps')
                plt.ylabel('Epsilon')
                plt.plot(epsilons)
                plt.show()
        else:
            plt.clf()
            plt.suptitle('Training...')
            plt.title(f'Current Average Reward : {averageOfLast100[-1]}')
            plt.xlabel('Episode')
            plt.ylabel('Rewards')
            plt.plot(rewards, '.b', label="Rewards")
            plt.plot(averageOfLast100, '-r', label="Average Rewards")
            plt.legend()
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def debug(self,log):
        if self.options.debug:
            print(log)

    def train(self):
        env = self.env
        observation_space = self.observation_space
        bestReward = 0
        bestAverageReward = 0
        rewards = []
        averageRewards = []
        epsilons = []
        for i in range(1, self.episodes):
            if isGymEnvLatest:
                state,info = env.reset()
            else:
                state = env.reset()
            state = np.reshape(state, [1, observation_space])
            totalRewardPerEpisode = 0
            steps = 0
            epsilons.append(self.epsilon)
            while True:
                action = self.getAction(state)
                if isGymEnvLatest:
                    nextState, reward, terminated,truncated, _ = env.step(action)
                    done = terminated or truncated
                else:
                    nextState, reward, done, _ = env.step(action)
                nextState = np.reshape(nextState, [1, observation_space])
                self.memory.add((state[0], action, reward, nextState[0], 1 - done))
                self.optimize()
                state = nextState
                totalRewardPerEpisode += reward
                
                diff = self.epsilonMax - self.epsilonMin
                decayed_epsilon = self.epsilonMin + diff * \
                    np.exp((-1 * self.iterations) / self.epsilonDecay)
                self.iterations += 1
                self.epsilon = max(self.epsilonMin, decayed_epsilon)
                # epsilons.append(self.epsilon)

                steps += 1
                if steps % self.updateFrequency:
                    self.syncWeights()

                if done:
                    rewards.append(totalRewardPerEpisode)
                    if totalRewardPerEpisode > bestReward:
                        bestReward = totalRewardPerEpisode

                    averageReward = np.mean(np.array(rewards)[-100:])
                    if averageReward > bestAverageReward:
                        bestAverageReward = averageReward
                    
                    
                    averageRewards.append(averageReward)
                    if self.options.showLiveResults:
                        self.plotResults(rewards, averageOfLast100=averageRewards)
                    elif self.options.logResults:
                        print('-'*80)
                        print(
                            f"\nEpisode {i} \
                            \nAverage Reward of last 100 {averageReward} \
                            \nBest Average Reward of last 100 {bestAverageReward} \
                            \nBest Reward {bestReward} \
                            \nCurrent Reward {totalRewardPerEpisode} \
                            \nEpsilon {self.epsilon}\n"
                        )
                    break
            
        self.plotResults(
            rewards, 
            averageRewards, 
            epsilons,
            done=True
        )
        self.saveRewards((rewards, averageRewards))
        self.results = (
            rewards,
            averageRewards,
            epsilons
        )
    
    def loadWeights(self, policyWeights, targetWeights):
        self.policyNetwork.load_state_dict(policyWeights)
        self.targetNetwork.load_state_dict(targetWeights)
    
    def greedy(self, timeSteps, ylim = None):
        env = self.env
        observation_space = self.observation_space
        rewards = []
        for i in range(timeSteps):
            if isGymEnvLatest:
                state,info = env.reset()
            else:
                state = env.reset()
            state = np.reshape(state, [1, observation_space])
            totalRewardPerEpisode = 0
            while True:
                action = self.getAction(state, train=False)
                if isGymEnvLatest:
                    nextState, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                else:
                    nextState, reward, done, _ = env.step(action)
                totalRewardPerEpisode += reward
                state = nextState
                self.debug(f"Reward:{reward}, {terminated=},{truncated=}")

                if done:
                    rewards.append(totalRewardPerEpisode)
                    print('-'*80)
                    print(
                        f"\nEpisode {i} \
                        \nCurrent Reward {totalRewardPerEpisode} "
                    )
                    break
        
        fig = plt.figure(1)
        fig.set_figwidth(12)
        fig.set_figheight(10)
        filePrefix = self.options.filePrefix
        title = 'Greedy - Rewards vs Episodes'
        if filePrefix:
            title += ' - ' + filePrefix
        plt.title(title)
        resultsPath = self.options.resultsPath
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        if ylim:
            plt.ylim(ylim)
        plt.plot(rewards, '-r', label="Rewards")
        plt.legend()
        if resultsPath:
            plt.savefig(f'{resultsPath}/images/{filePrefix}_{self.name}_Greedy_EpisodesVsRewards.png')
        plt.show()
        

class DuelingDQN(DQN):
    def __init__(self, env, hyperparams: Hyperparams, nnModel, options: Options = {}):
        super().__init__(env, hyperparams, nnModel, options)
        self.name = type(self).__name__
    
    def syncWeights(self):
        policyWeights = self.policyNetwork.state_dict()
        targetWeights = self.targetNetwork.state_dict()
        for key in policyWeights:
            targetWeights[key] = policyWeights[key]*self.tau + targetWeights[key]*(1-self.tau)
        self.targetNetwork.load_state_dict(targetWeights)

    def optimize(self):
        batchSize = self.batchSize
        if len(self.memory) > batchSize:
            minibatch = np.array(self.memory.sample(batchSize))
            states = minibatch[:, 0].tolist()
            actions = minibatch[:, 1].tolist()
            rewards = minibatch[:, 2].tolist()
            nextStates = minibatch[:, 3].tolist()
            dones = minibatch[:, 4].tolist()

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            nextStates = torch.tensor(
                nextStates, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)
            indices = np.arange(batchSize, dtype=np.int64)

            qValues = self.policyNetwork(states)
            qDotValues = None
            with torch.no_grad():
                qDotValues = self.targetNetwork(nextStates)

            predictedValues = qValues[indices, actions]
            predictedQDotValues = torch.max(qDotValues, dim=1)[0]

            targetValues = (1-self.learningRate)*predictedValues+(self.learningRate)*(rewards + self.discountFactor * predictedQDotValues * dones)

            loss = self.policyNetwork.loss(targetValues, predictedValues)
            self.policyNetwork.optimizer.zero_grad()
            loss.backward()
            self.policyNetwork.optimizer.step()
    
class DoubleDQN(DQN):
    def __init__(self, envInfo: EnvInfo, hyperparams: Hyperparams, nnModel, options: Options = {}):
        super().__init__(envInfo, hyperparams, nnModel, options)
        self.name = type(self).__name__

           
    def syncWeights(self):
        policyWeights = self.policyNetwork.state_dict()
        targetWeights = self.targetNetwork.state_dict()
        for key in policyWeights:
            targetWeights[key] = policyWeights[key]*self.tau + targetWeights[key]*(1-self.tau)
        self.targetNetwork.load_state_dict(targetWeights)

    def optimize(self):
        batchSize = self.batchSize
        if len(self.memory) > batchSize:
            minibatch = np.array(self.memory.sample(batchSize))
            states = minibatch[:, 0].tolist()
            actions = minibatch[:, 1].tolist()
            rewards = minibatch[:, 2].tolist()
            nextStates = minibatch[:, 3].tolist()
            dones = minibatch[:, 4].tolist()

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            nextStates = torch.tensor(
                nextStates, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)
            indices = np.arange(batchSize, dtype=np.int64)

            qValues = self.policyNetwork(states)

            qDotValues = None
            qNewValues = None
            with torch.no_grad():
                qNewValues =  self.policyNetwork(nextStates)
                qDotValues = self.targetNetwork(nextStates)
            
            maxActions = torch.argmax(qDotValues,dim=1)
            normalValue = qValues[indices, actions]
            predictedValues = qNewValues[indices, maxActions]
            targetValues = rewards + self.discountFactor * predictedValues * dones
            loss = self.policyNetwork.loss(targetValues, normalValue)
            self.policyNetwork.optimizer.zero_grad()
            loss.backward()
            self.policyNetwork.optimizer.step()
            