# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
from typing import Tuple, List, Union
import yaml
import random
from functools import reduce
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    ShuffleSplit,
    StratifiedShuffleSplit,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DataParallel as DP
import torch.distributed as dist

from tokenizer import DNATokenizer
from dataset import UTRDataset
from model import PerformerModel


def _argparse() -> argparse.Namespace:
    """
    Returns:
        argparse.Namespace: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="config file path")
    parser.add_argument("--data", required=True, type=str, help="path to csv file data")
    args = parser.parse_args()

    return args


def _parse_config(cfg_path: str) -> dict:
    """Load and return yaml format config

    Args:
        cfg_path (str): yaml config file path

    Returns:
        config (dict): config dict
    """

    with open(cfg_path) as f:
        config = yaml.safe_load(f.read())

    return config


def load_data(data_path: str) -> Tuple[List, List]:
    """
    Args:
        data_path (str): path to csv file

    Returns:
        Tuple[List, List]: Each list of [data,label]
    """
    full_data = pd.read_csv(data_path, index_col=0)
    data = full_data.loc[:, ["5UTR", "CDS", "3UTR"]].values
    label = full_data["label"].values
    return (data, label)


def train(
    cfg: yaml,
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    optimizer: torch.optim,
) -> None:
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.train_bs, shuffle=True, num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.train.val_bs, shuffle=False, num_workers=2
    )


if __name__ == "__main__":

    args = _argparse()
    cfg = _parse_config(args.cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpu = len(cfg.gpus)

    data, label = load_data(args.data_path)
    train_data, val_data, train_label, val_label = train_test_split(
        data, label, test_size=0.2, random_state=cfg.seed
    )
    train_dataset = UTRDataset(train_data, train_label)
    val_dataset = UTRDataset(val_data, val_label)

    model = PerformerModel(cfg.models)
    model.to(device)
    if num_gpu > 1:
        model = DP(model)

    # optimizer
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
