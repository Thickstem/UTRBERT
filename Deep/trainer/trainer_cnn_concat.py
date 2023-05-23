# -*- coding: utf-8 -*-
import os
import json
import sys

sys.path.append("..")
from logging import getLogger, config
from typing import Tuple, Union
import yaml
from tqdm import tqdm
import hydra
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DataParallel as DP
import torch.distributed as dist

from dataset import UTRDataset_CNN_CONCAT
from models.model_cnn import MRNA_CNN_CONCAT
from utils import metrics


def load_data(data_path: str) -> Tuple[list, list]:
    """
    Args:
        data_path (str): path to csv file

    Returns:
        Tuple[List, List]: Each list of [data,label]
    """
    full_data = pd.read_csv(data_path, index_col=0)
    seq_data = full_data.loc["seq"].values
    feat_data = full_data.iloc[1:-1].values
    label = full_data["te"].values
    return (seq_data, feat_data), label


def validation(
    cfg: yaml, model: nn.Module, val_dataset: Dataset, scaler: StandardScaler
):
    """

    Args:
        cfg (yaml): _description_
        model (nn.Module): _description_
        val_dataset (Dataset): _description_
        scaler (StandardScaler): _description_

    Returns:
        _type_: _description_
    """
    device = model.device
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.train.val_bs, shuffle=False, drop_last=True
    )
    loss_fn = MSELoss()
    model.eval()
    running_loss = 0
    eval_steps = 0
    preds = None
    for data, labels in val_dataloader:
        eval_steps += 1
        seq_data, feat_data = data[0].to(device), data[1].to(
            device
        )  # [input_ids,attention_masks]
        labels = labels.to(device)

        logits = model(seq_data, feat_data)
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

    scores = metrics(preds.view(-1), out_labels)
    scores["val_loss"] = eval_loss

    return scores


def train(
    cfg: yaml,
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    optimizer: torch.optim,
) -> None:
    device = model.device
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.train_bs, shuffle=True, drop_last=True
    )
    loss_fn = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=cfg.scheduler_step)

    for epoch in range(cfg.train.epoch):
        model.train()
        running_loss = 0.0
        train_steps = 0
        for data, labels in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            seq_data, feat_data = data[0].to(device), data[1].to(
                device
            )  # [input_ids,attention_masks]
            labels = labels.to(device)
            logits = model(seq_data, feat_data)

            logits.view(-1)
            loss = loss_fn(logits.view(-1), labels.view(-1))
            if len(cfg.gpus) > 1:
                loss = loss.mean()
            if cfg.train.grad_acc > 1:
                loss = loss / cfg.train.grad_acc

            loss.backward()

            if (train_steps + 1) % cfg.train.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            train_steps += 1

        epoch_loss = running_loss / train_steps
        wandb.log(
            {
                "train_loss": epoch_loss,
                "learning rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )
        print(f"Epoch {epoch} loss:{epoch_loss}")
        scheduler.step()

        if (epoch + 1) % cfg.train.val_epoch == 0:
            scores = validation(cfg, model, val_dataset, scaler=train_dataset.scaler)
            for key, value in scores.items():
                wandb.log({key: value}, step=epoch)
                print(f"{key}:{value:.4f}")

    torch.save(model.module.state_dict(), os.path.join(cfg.result_dir, "latest.pth"))


@hydra.main(confing_path="/home/ksuga/UTRBERT/Deep/configs", config_name="cnn_concat")
def main(cfg: hydra):
    logger = getLogger("Log")
    logger.info(cfg)

    wandb.init(
        project=cfg.project_name,
        group=cfg.exp_name,
        config=cfg,
        mode=cfg.wandb_mode,
    )

    cfg.result_dir = os.path.join("results", cfg.result_dir)
    os.makedirs("results", exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpu = len(cfg.gpus)

    logger.info(f"loading data from {cfg.data_path}")
    data, label = load_data(cfg.train.data_path)  # data=(seq_data,feat_data)
    train_data, val_data, train_label, val_label = train_test_split(
        data, label, test_size=0.2, random_state=cfg.seed
    )

    print("Creating train dataset ...")
    train_dataset = UTRDataset_CNN_CONCAT(train_data[0], train_data[1], train_label)
    print("Creating val dataset...")
    val_dataset = UTRDataset_CNN_CONCAT(val_data[0], val_data[0], val_label)

    model = MRNA_CNN_CONCAT(cfg)
    model.to(device)
    if num_gpu > 1:
        model = DP(model)

    # optimizer
    optimizer = Adam(model.parameters(), lr=float(cfg.lr))

    train(cfg, model, train_dataset, val_dataset, optimizer)


if __name__ == "__main__":
    main()
