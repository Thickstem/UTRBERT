# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np

from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tokenizer import DNATokenizer
from dataset import UTRData
from model import PerformerModel

def _argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dim","required")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = _argparse()
