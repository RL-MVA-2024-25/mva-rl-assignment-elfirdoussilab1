# In this file, we will implement the ML models used for our decision making
import torch
import torch.nn as nn
import torch.nn.functional as F

class simple_relu_nn(nn.Module):
    def __init__(self, hidden_neurons):
        super().__init__(self)
        self.fc1 = nn.Linear(6, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, 4)

    def forward(self, s):
        out = F.relu(self.fc1(s))
        out = self.fc2(out)
        return out
