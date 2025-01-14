import numpy as np
#from tqdm import tqdm
from data import *
import torch
from model import *
from data import *
from utils import greedy_action

class ProjectAgent:
    def __init__(self):
        self.model = torch.nn.Sequential(nn.Linear(6, 256),
                          nn.ReLU(),
                          nn.Linear(256, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 32),
                          nn.ReLU(), 
                          nn.Linear(32, 4))
        
        # self.model = torch.nn.Sequential(nn.Linear(6, 256),
        #                 nn.ReLU(),
        #                 nn.Linear(256, 512),
        #                 nn.ReLU(),
        #                 nn.Linear(512, 512),
        #                 nn.ReLU(),
        #                 nn.Linear(512, 256),
        #                 nn.ReLU(), 
        #                 nn.Linear(256, 4))
        

    def act(self, observation, eps = 0, use_random=False): # epsilon-greedy action
        # # Epsilon-greedy action
        # r = random.random()
        # if r < eps:
        #     k = np.random.choice(np.arange(0, 4))
        # else:
        #     k = greedy_action(self.model, observation)
        # return k
        return greedy_action(self.model, observation)

    def save(self, path = "model_dqn.pth"):
        # Save the model
        torch.save(self.model.state_dict(), path)

    def load(self, path = "models/model_dqn_best.pth"):
        # Load the RF model
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only = True))
