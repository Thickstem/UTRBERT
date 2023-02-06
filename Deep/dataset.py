import os
import json
import copy
from logging import getLogger
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from utils import seq_n_padding, onehot_encode

logger = getLogger("Log").getChild("dataset")


def create_input_features(
    cfg,
    phase,
    samples,
    labels,
    tokenizer,
    max_length,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    cached_feature_file = os.path.join(
        cfg.data_dir,
        "cached_{}_{}_{}.pt".format(
            os.path.basename(cfg.data).split(".")[0], cfg.dataset.max_length, phase
        ),
    )
    if os.path.exists(cached_feature_file):
        logger.info("Loading cached dataset ...")
        features = torch.load(cached_feature_file)

    else:
        features = []
        for i, (sample, label) in tqdm(
            enumerate(zip(samples, labels)), desc="Creating features..."
        ):
            sample = " ".join(
                sample
            )  # convert ["fiveprime","cds","threeprime"]→ ["fiveprime cds threeprime"]
            tokens = tokenizer.encode_plus(
                sample,
                add_special_tokens=True,
            )
            input_ids, attention_mask, token_type_ids = (
                tokens["input_ids"],
                tokens["attention_mask"],
                tokens["token_type_ids"],
            )
            padding_length = max_length - len(input_ids)

            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
                token_type_ids = (
                    [pad_token_segment_id] * padding_length
                ) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                token_type_ids = token_type_ids + (
                    [pad_token_segment_id] * padding_length
                )

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label,
                )
            )
        torch.save(features, cached_feature_file)

    return features


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class UTRDataset(Dataset):
    def __init__(self, cfg, data, label, tokenizer, phase="train"):
        "Input format should be id-converted sequences"
        super().__init__()
        self.data = data
        self.label = label
        self.phase = phase
        self.features = create_input_features(
            cfg,
            self.phase,
            self.data,
            self.label,
            tokenizer,
            max_length=cfg.dataset.max_length + 2,
            pad_on_left=cfg.dataset.pad_on_left,
            pad_token=cfg.dataset.pad_token,
            pad_token_segment_id=cfg.dataset.pad_token_segment_id,
            mask_padding_with_zero=cfg.dataset.mask_padding_with_zero,
        )
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in self.features], dtype=torch.long
        )
        self.all_attention_mask = torch.tensor(
            [f.attention_mask for f in self.features],
            dtype=torch.bool,  # for performer input
        )

        all_labels = np.array([f.label for f in self.features], dtype=float)
        if phase == "val":
            self.scaler = StandardScaler().fit(all_labels.reshape(-1, 1))
        all_labels = (
            StandardScaler().fit_transform(all_labels.reshape(-1, 1)).reshape(-1)
        )
        self.all_labels = torch.tensor(all_labels, dtype=torch.float)

    def __getitem__(self, index):
        "output format should be id-converted sequences"
        inputs = (
            self.all_input_ids[index],
            self.all_attention_mask[index],
            # self.all_token_type_ids[index],
        )
        return inputs, self.all_labels[index]

    def __len__(self):
        return self.data.shape[0]


class UTRDataset_CNN(Dataset):
    def __init__(self, data, label, phase):
        super().__init__()
        self.data = data
        self.label = label
        if phase == "train":
            self.scaler = StandardScaler().fit(self.label.reshape(-1, 1))
            self.label = self.scaler.transform(self.label.reshape(-1, 1))

        self.label = torch.tensor(self.label, dtype=torch.float)

    def __getitem__(self, index):
        seqs = self.data[index]
        seq = seq_n_padding(seqs)
        onehot_seq = onehot_encode(seq)
        return torch.tensor(onehot_seq, dtype=torch.float), self.label[index]

    def __len__(self):
        return self.data.shape[0]
