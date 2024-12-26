# In this file, we will implement the function to generate our dataset of states, actions and rewards
# by interacting with the environment
import numpy as np
from tqdm import tqdm

# Collecting data samples
def create_dataset(env, horizon, disable_tqdm=False):
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
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
        np.savez_compressed(path, *dataset)
    else:
        S = dataset[0]
        n = len(S)
        np.savez_compressed(f'data_{n}.npz', *dataset)

def load_dataset(path):
    dataset = np.load('arrays.npz')
    return dataset