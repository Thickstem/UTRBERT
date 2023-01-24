# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
from typing import Tuple, Union
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


def load_data(data_path: str) -> Tuple[list, list]:
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


def val(cfg: yaml, model: nn.Module, val_dataset: Dataset):

    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.train.val_bs, shuffle=False, num_workers=2
    )
    model.eval()
    for data, labels in val_dataloader:
        data, labels = data.to(device), labels.to(device)

        logits = model(data)
        loss = MSELoss(logits, labels)
        running_loss += loss.item()

    eval_loss = running_loss // len(val_dataloader)

    return eval_loss


def train(
    cfg: yaml,
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    optimizer: torch.optim,
) -> None:
    cfg = cfg.train
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train_bs, shuffle=True, num_workers=2
    )

    for epoch in range(cfg.epoch):
        model.train()
        running_loss = 0.0
        for step, (data, labels) in enumerate(train_dataloader):
            data, labels = data.to(device), labels.to(device)

            logits = model(data)
            loss = MSELoss(logits, label)
            if len(cfg.gpus) > 1:
                loss = loss.mean()
            if cfg.grad_acc > 1:
                loss = loss / cfg.grad_acc

            loss.backward()

            if (step + 1) // cfg.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        model.zero_grad()

        if epoch % cfg.val_epoch == 0:
            eval_loss = val(cfg, model, val_dataset)


if __name__ == "__main__":

    args = _argparse()
    cfg = _parse_config(args.cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpu = len(cfg.gpus)

    tokenizer = DNATokenizer(vocab_file=cfg.vocab_file)

    data, label = load_data(args.data_path)
    train_data, val_data, train_label, val_label = train_test_split(
        data, label, test_size=0.2, random_state=cfg.seed
    )
    train_dataset = UTRDataset(cfg, train_data, train_label, tokenizer)
    val_dataset = UTRDataset(cfg, val_data, val_label, tokenizer)

    model = PerformerModel(cfg.models)
    model.to(device)
    if num_gpu > 1:
        model = DP(model)

    # optimizer
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
