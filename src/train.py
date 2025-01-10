from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from data import *
import joblib # To save and load Random Forest Regressor
import random
from sklearn.linear_model import LinearRegression
from model import *
import matplotlib.pyplot as plt
import torch.nn as nn

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device)

class ProjectAgent:
    def __init__(self):
        self.Q = None
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])

    def act(self, observation, use_random=False):
        # We will use the Greedy action that maximizes the utility function
        Q = model(torch.tensor(observation, dtype = torch.float).to(device))
        # Epsilon-greedy action
        r = random.random()
        if r < 0.1:
            k = np.random.choice(np.arange(0, 4))
            return k
        return np.argmax(Q.cpu().detach().numpy())

    def save(self, path):
        # Save the model
        torch.save(model.state_dict(), "model_dqn.pth")

    def load(self, path = "models/model_dqn.pth"):
        # Load the RF model
        self.model = simple_relu_nn(config['hidden_neurones']).to(device)
        self.model.load_state_dict(torch.load(path))

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma) # does: R + gamma * (1 - D) * QYmax
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                #action = env.action_space.sample()
                action = np.random.choice(np.arange(0, 4))
            else:
                action = self.act(state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward # We don't do gamma discounting, because we know that the number 
                                        # of steps (horizon) is finite.

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)),# should be named: memory length 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return
    
# Config
config = {'nb_actions': 4,
          'hidden_neurones': 128,
          'learning_rate': 0.001,
          'gamma': 0.9,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 1000,
          'epsilon_delay_decay': 20,
          'batch_size': 64}

if __name__ == "__main__":
    # Model
    model = torch.nn.Sequential(nn.Linear(6, config['hidden_neurones']),
                          nn.ReLU(),
                          nn.Linear(config['hidden_neurones'], 4)).to(device)

    # Train agent
    agent = ProjectAgent()
    scores = agent.train(env, 200)
    fig, ax = plt.subplots()
    ax.plot(scores)
    fig.savefig("plot_dqn.pdf")

