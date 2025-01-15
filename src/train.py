from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from data import *
import torch
import torch.nn as nn
from data import *
from utils import greedy_action
from copy import deepcopy
#import matplotlib.pyplot as plt
#from fast_env import FastHIVPatient
from evaluate import evaluate_HIV

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device)

# Training configuration
config = {'nb_actions': 4,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'buffer_size': 100000,
            'epsilon_min': 0.01,
            'epsilon_max': 1.,
            'epsilon_decay_period': 10000,
            'epsilon_delay_decay': 100,
            'batch_size': 512,
            'gradient_steps': 5,
            'update_target_strategy': 'replace', # or 'ema'
            'update_target_freq': 50,
            'update_target_tau': 0.005,
            'criterion': torch.nn.SmoothL1Loss(),
            'monitoring_nb_trials': 50,
            'strategy': 'random'} # should be in ['random', 'fix']

# 'random' strategy is the one that leads to the best performance: it consists on training the agent of the random env
# 'fix' strategy consists on training the agent on the fixed Patient: leads to a score of 5

class ProjectAgent:
    def __init__(self):
        # Model: Deep Neural Network with 5 hidden layers of different sizes
        self.model = nn.Sequential(nn.Linear(6, 256),
                            nn.ReLU(),
                            nn.Linear(256, 512),
                            nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(), 
                            nn.Linear(128, 4)).to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.ddqn = False # Double DQN

        # Config
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.train_strategy = config['strategy']

    def act(self, observation, use_random=False):
        # Greedy action
        return greedy_action(self.model, observation)

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)

            if self.ddqn:
                best_actions = torch.argmax(self.model(Y), dim=1) # shape (B, )
                QYmax = self.target_model(Y)[torch.arange(self.batch_size), best_actions].detach()
            else: # Simple DQN with target model
                QYmax = self.target_model(Y).max(1)[0].detach() # max(Qt[new_state])

            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  

    def save(self, path = "model_dqn.pth", path_target = "target_model.pth"):
        # Save the model
        torch.save(self.model.state_dict(), path)
        torch.save(self.target_model.state_dict(), path_target)

    def load(self, path = "model_dqn_best_1.pth"): # Best Model gotten so far
        # Load the DQN model
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only = True))
    
    def train(self, max_episode):
        # Environment
        if self.train_strategy == 'random':
            env = TimeLimit(
                env=HIVPatient(domain_randomization=True), 
                #env = FastHIVPatient(domain_randomization=True),
                max_episode_steps=200
            )
        else: # fixed!
            env = TimeLimit(
                env=HIVPatient(domain_randomization=False), 
                #env = FastHIVPatient(domain_randomization=True),
                max_episode_steps=200
            )

        episode_return = []
        best_eval = 0
        count = 0
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
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema': # ema = exp moving average
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Evaluate the current model on the environment for an episode
                eval = evaluate_HIV(agent=self, nb_episode=1)
                # If the evaluation score improved, save this checkpoint
                if best_eval < eval :
                    best_eval = eval
                    count += 1
                    path = f'model_dqn_{count}.pth'
                    self.save(path)
                    print("New Model saved !")
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      ", best ", '{:4.1f}'.format(best_eval),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return

if __name__ == "__main__":
    # Total number of episodes
    n_epsides = 2000

    # Agent
    agent = ProjectAgent()
    print("Training...")
    scores = agent.train(n_epsides)

    print("Finished training.")
    # Plotting results
    # fig, ax = plt.subplots()
    # ax.plot(scores)
    # path = "train_plot.pdf"
    # fig.savefig(path)
    # print("Plot saved")