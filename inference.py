#!/usr/bin/env python

import os
import glob
import numpy as np
import torch
from torch.nn import functional as F
from utils import load_csv, create_train_test_split
from model import AnomalyDetector

device = "cuda"
window_size = 30
stride = 5
cwd = os.path.dirname(os.path.abspath(__file__))

# load data
create_train_test_split()
test_csv_files = glob(os.path.join(cwd, "data/test/*"))
test_dfs = load_csv(test_csv_files)
print('Finished loading data')

# load model
model = AnomalyDetector().to(device)
model.load_state_dict(torch.load("./checkpoints/transformer-anomaly-annotator-best.pt"))
model.eval()
print('Successfully loaded model')

# train
epochs = 2
batch_size = 512
max_steps = epochs * (sum([len(df) for df in test_dfs]) / (window_size * batch_size))

itr = 1
losses = []
while itr <= max_steps:
    dfs = [
        test_dfs[i] for i in np.random.randint(0, len(test_dfs), size=(batch_size,))
    ]
    idxs = [np.random.randint(0, len(df) - window_size) for df in dfs]
    data = np.stack([df[idx : idx + window_size].values for df, idx in zip(dfs, idxs)])
    x = torch.from_numpy(data).to(device).float()

    # forward
    out = model(x)
    # calc loss
    loss = F.mse_loss(out, x)

    # metrics
    losses.append(loss.item())

    itr += 1

print(f"Average test loss: {np.mean(losses)}")
