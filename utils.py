#!/usr/bin/env python

import os
from typing import List
import shutil
import numpy as np
import pandas as pd

def preprocess(df: pd.DataFrame, window_size=30):
    df = df.iloc[: df.shape[0] // window_size * window_size]
    # fill na
    df = df.interpolate().bfill()
    # TODO: replace it with batch norm
    df = (df - df.mean(axis=0)) / (df.std(axis=0) + 1e-6)  # prevent 0/0
    df.pop("timestamp")
    return df


def load_csv(csv_files: List[str]):
    dfs: List[pd.DataFrame] = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        hz = 1e6 / df["timestamp"].diff().dropna().mean()
        # only include file with 10Hz freq logs
        if hz < 9 or hz > 11:
            continue
        df = preprocess(df)
        dfs.append(df)
    return dfs


def create_train_test_split():
    cwd = os.path.dirname(os.path.abspath(__file__))

    train_dir = os.path.join(cwd, "data/train")
    val_dir = os.path.join(cwd, "data/val")
    test_dir = os.path.join(cwd, "data/test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
        print('train, val and test dirs already exist')
        return

    print('Creating train, val and test dirs...')

    data_dir = os.path.join(cwd, "data/csv_files")
    csv_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")
    ]

    np.random.shuffle(csv_files)

    train_csv_files = csv_files[: int(0.70 * len(csv_files))]
    val_csv_files = csv_files[int(0.70 * len(csv_files)) : int(0.85 * len(csv_files))]
    test_csv_files = csv_files[int(0.85 * len(csv_files)) :]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for f in train_csv_files:
        shutil.copy2(f, train_dir)
    for f in val_csv_files:
        shutil.copy2(f, val_dir)
    for f in test_csv_files:
        shutil.copy2(f, test_dir)
    print('Successfully created train, val and test dirs')
