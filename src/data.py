# In this file, we will implement the function to generate our dataset of states, actions and rewards
# by interacting with the environment, as well as the ReplayBuffer for training DQNs
import numpy as np
import random
import torch

# Collecting data samples
def create_dataset(env, horizon, disable_tqdm=False):
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in range(horizon):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        #dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D

def save_dataset(dataset, path = None):
    # path should be of the form: name.npz
    if path is not None:
        np.savez_compressed(path, S=dataset[0], A=dataset[1], R=dataset[2], S2=dataset[3], D=dataset[4])
    else:
        S = dataset[0]
        n = len(S)
        np.savez_compressed(f'data_{n}.npz', S=dataset[0], A=dataset[1], R=dataset[2], S2=dataset[3], D=dataset[4])

def load_dataset(path):
    dataset = np.load(path)
    return dataset

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


# if __name__ == "__main__":
#     env = TimeLimit(
#         env=HIVPatient(domain_randomization=False), max_episode_steps=200
#     )
#     horizon = int(1e6)
#     dataset = create_dataset(env, horizon)
#     save_dataset(dataset, path = 'data_1M.npz')
