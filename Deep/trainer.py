# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
from logging import getLogger, config
from attrdict import AttrDict
from typing import Tuple, Union
import yaml
import random
from functools import reduce
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

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
from utils import metrics


def _argparse() -> argparse.Namespace:
    """
    Returns:
        argparse.Namespace: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="config file path")
    parser.add_argument("--logger_cfg", default="./configs/log_config.json")
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
    config = AttrDict(config)
    return config


def load_data(data_path: str) -> Tuple[list, list]:
    """
    Args:
        data_path (str): path to csv file

    Returns:
        Tuple[List, List]: Each list of [data,label]
    """
    full_data = pd.read_csv(data_path, index_col=0)
    data = full_data.loc[:, ["fiveprime", "cds", "threeprime"]].values
    label = full_data["te"].values
    return (data, label)


def val(cfg: yaml, model: nn.Module, val_dataset: Dataset):

    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.train.val_bs, shuffle=False, num_workers=2
    )
    model.eval()
    eval_steps = 0
    for data, labels in val_dataloader:
        eval_steps += 1
        data, labels = data.to(device), labels.to(device)

        logits = model(data)
        loss = MSELoss(logits, labels)
        running_loss += loss.item()
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_labels = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_labels = np.append(out_labels, labels.detach().cpu().numpy(), axis=0)

    eval_loss = running_loss // eval_steps
    scores = metrics(preds, out_labels)
    scores["loss"] = eval_loss

    return scores


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
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=cfg.lr,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9,
    )

    for epoch in range(cfg.epoch):
        model.train()
        running_loss = 0.0
        train_steps = 0
        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)

            logits = model(data)
            loss = MSELoss(logits, label)
            if len(cfg.gpus) > 1:
                loss = loss.mean()
            if cfg.grad_acc > 1:
                loss = loss / cfg.grad_acc

            loss.backward()

            if (train_steps + 1) // cfg.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            train_steps += 1

        epoch_loss = running_loss / train_steps
        scheduler.step()
        model.zero_grad()
        wandb.log({"train_loss": epoch_loss}, step=epoch)

        if epoch % cfg.val_epoch == 0:
            scores = val(cfg, model, val_dataset)
            for key, val in scores.items():
                wandb.log({key: val}, step=epoch)

    torch.save(model.module.state_dict(), cfg.result_dir)


if __name__ == "__main__":

    args = _argparse()

    # Setting logger
    with open(args.logger_cfg, "r") as f:
        log_conf = json.load(f)
    config.dictConfig(log_conf)
    logger = getLogger(__name__)

    cfg = _parse_config(args.cfg)
    logger.info(cfg)
    wandb.init(
        name=f"{os.path.basename(cfg.result_dir)}", project="mrna_full_dev", config=cfg
    )

    os.makedirs(cfg.result_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpu = len(cfg.gpus)

    tokenizer = DNATokenizer(
        vocab_file=cfg.dataset.vocab_file, max_len=cfg.dataset.max_length
    )  # for [CLS],[SEP] tokens

    data, label = load_data(cfg.data)
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

    train(cfg, model, train_dataset, val_dataset, optimizer)
