# Util functions
import torch

def greedy_action(model, state):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = model(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
