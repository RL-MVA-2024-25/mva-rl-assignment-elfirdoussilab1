from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from data import *
import joblib # To save and load Random Forest Regressor

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.Q = None

    def act(self, observation, use_random=False):
        # We will use the Greedy action that maximizes the utility function
        Qsa = []
        # Looping over actions
        for a in range(4):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        return np.argmax(Qsa)

    def save(self, path):
        # Save the model
        joblib.dump(self.Q, path)

    def load(self, path = "random_forest_regressor.pkl"):
        # Load the RF model
        loaded_model = joblib.load(path)
        self.Q = loaded_model


def train(model = "rf", data_dir = "data.npz"):
    # Adding an arg parser: Random Forest Or Neural Network
    dataset = load_dataset(data_dir)
    S, A, R, S2, D = dataset.values()

    if model == "rf":
        rf = RandomForestRegressor()

def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
    nb_samples = S.shape[0]
    Qfunctions = []
    SA = np.append(S,A,axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter==0:
            value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((S.shape[0],1))
                S2A2 = np.append(S2,A2,axis=1)
                Q2[:,a2] = Qfunctions[-1].predict(S2A2) # Using the previous q function to predict next one
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*(1-D)*max_Q2
        Q = RandomForestRegressor()
        Q.fit(SA,value)
        Qfunctions.append(Q)
    return Qfunctions

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--data_dir", type=Path, required=True, help="Path to image directory"
#     )
#     parser.add_argument("--save_dir", type=Path, default= None)
#     #parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--model", type=str, default="rf")
#     args = parser.parse_args()

#     train(args.model, args.data_dir)
