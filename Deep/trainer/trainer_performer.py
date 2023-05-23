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
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
from Deep.models.model_performer import PerformerModel
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


def validation(cfg: yaml, model: nn.Module, val_dataset: Dataset):
    """

    Args:
        cfg (yaml): cfg.train
        model (nn.Module): _description_
        val_dataset (Dataset): _description_

    Returns:
        _type_: _description_
    """
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.train.val_bs, shuffle=False, num_workers=2
    )
    loss_fn = MSELoss()
    model.eval()
    running_loss = 0
    eval_steps = 0
    preds = None
    for data, labels in val_dataloader:
        eval_steps += 1
        data = tuple(d.to(device) for d in data)  # [input_ids,attention_masks]
        labels = labels.to(device)

        logits = model(data[0], mask=data[1])
        loss = loss_fn(logits.view(-1), labels.view(-1))
        if len(cfg.gpus) > 1:
            loss = loss.mean()
        running_loss += loss.item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_labels = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_labels = np.append(out_labels, labels.detach().cpu().numpy(), axis=0)

    eval_loss = running_loss / eval_steps
    preds = val_dataset.scaler.inverse_transform(preds).reshape(-1)
    out_labels = val_dataset.scaler.inverse_transform(
        out_labels.reshape(-1, 1)
    ).reshape(-1)
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
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.train_bs, shuffle=True, num_workers=2
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    for epoch in range(cfg.train.epoch):
        model.train()
        running_loss = 0.0
        train_steps = 0
        for data, labels in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            data = tuple(d.to(device) for d in data)  # [input_ids,attention_masks]
            labels = labels.to(device)

            logits = model(data[0], mask=data[1])
            loss = loss_fn(logits.view(-1), labels.view(-1))
            if len(cfg.gpus) > 1:
                loss = loss.mean()
            if cfg.train.grad_acc > 1:
                loss = loss / cfg.train.grad_acc

            loss.backward()

            if (train_steps + 1) % cfg.train.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            train_steps += 1

        epoch_loss = running_loss / train_steps
        wandb.log(
            {"train_loss": epoch_loss, "learning rate": scheduler.get_lr()}, step=epoch
        )
        print(f"Epoch {epoch} loss:{epoch_loss}")

        if epoch % cfg.train.val_epoch == 0:
            scores = validation(cfg, model, val_dataset)
            for key, value in scores.items():
                wandb.log({key: value}, step=epoch)
                print(f"{key}:{value:.4f}")

    torch.save(model.module.state_dict(), cfg.result_dir)


if __name__ == "__main__":
    args = _argparse()
    cfg = _parse_config(args.cfg)

    # Setting logger
    with open(args.logger_cfg, "r") as f:
        log_conf = json.load(f)
    config.dictConfig(log_conf)
    logger = getLogger("Log")
    logger.info(cfg)

    wandb.init(
        name=f"{os.path.basename(cfg.result_dir)}", project="mrna_full", config=cfg
    )

    os.makedirs(cfg.result_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpu = len(cfg.gpus)

    tokenizer = DNATokenizer(
        vocab_file=cfg.dataset.vocab_file, max_len=cfg.dataset.max_length
    )  # for [CLS],[SEP] tokens

    logger.info(f"loading data from {cfg.data}")
    data, label = load_data(cfg.data)
    train_data, val_data, train_label, val_label = train_test_split(
        data, label, test_size=0.2, random_state=cfg.seed
    )

    print("Creating train dataset ...")
    train_dataset = UTRDataset(cfg, train_data, train_label, tokenizer, phase="train")
    print("Creating val dataset...")
    val_dataset = UTRDataset(cfg, val_data, val_label, tokenizer, phase="val")

    model = PerformerModel(cfg.models)
    model.to(device)
    if num_gpu > 1:
        model = DP(model)

    # optimizer
    optimizer = Adam(model.parameters(), lr=float(cfg.train.lr))

    train(cfg, model, train_dataset, val_dataset, optimizer)
