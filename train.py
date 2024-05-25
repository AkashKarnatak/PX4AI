#!/usr/bin/env python

import os
from glob import glob
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import load_csv, create_train_test_split
from model import AnomalyDetector

device = "cuda"
window_size = 30
stride = 5
cwd = os.path.dirname(os.path.abspath(__file__))

# load data
create_train_test_split()
train_csv_files = glob(os.path.join(cwd, "data/train/*"))
train_dfs = load_csv(train_csv_files)
print('Finished loading data')

# load model
model = AnomalyDetector().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# train
epochs = 40
batch_size = 512
max_steps = epochs * (sum([len(df) for df in train_dfs]) / (window_size * batch_size))

writer = SummaryWriter()
itr = 1
last_loss = float("inf")
losses = []
while itr <= max_steps:
    dfs = [
        train_dfs[i] for i in np.random.randint(0, len(train_dfs), size=(batch_size,))
    ]
    idxs = [np.random.randint(0, len(df) - window_size) for df in dfs]
    data = np.stack([df[idx : idx + window_size].values for df, idx in zip(dfs, idxs)])
    x = torch.from_numpy(data).to(device).float()

    # forward
    out = model(x)
    # calc loss
    loss = F.mse_loss(out, x)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # metrics
    losses.append(loss.item())
    writer.add_scalar("Loss/step", loss, itr)
    print(f"Step: {itr}, Loss: {loss.item()}")

    # save
    if itr % 100 == 0 and loss.item() < last_loss:
        torch.save(
            model.state_dict(),
            os.path.join(cwd, "checkpoints/transformer-anomaly-annotator-best.pt"),
        )
        last_loss = loss.item()

    itr += 1

torch.save(
    model.state_dict(), os.path.join(cwd, "checkpoints/transformer-anomaly-annotator-latest.pt")
)
