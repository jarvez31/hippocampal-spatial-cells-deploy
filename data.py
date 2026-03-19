import numpy as np
import pandas as pd
import os
import pickle


def load_trajectory(env: list, data_dir: str):
    filepath = os.path.join(data_dir, env)
    if filepath.endswith(".pk1"):
        with open(filepath, "rb") as f:
            d = pickle.load(f)
        x = np.asarray(d['x'])
        y = np.asarray(d['y'])
        z = np.zeros(len(x))
        return np.column_stack((x, y, z))
    else:
        data = pd.read_csv(filepath, header=None)
        return np.asarray(data)