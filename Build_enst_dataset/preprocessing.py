import os
import argparse
import numpy as np
import pandas as pd


def _argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_db", required=True, type=str)
    parser.add_argument("--te_data", required=True, type=str)
    parser.add_argument("--save", required=True, type=str)

    args = parser.parse_args()
    return args


def seq2mer(seq, kmer=3):
    mers = []
    for i in range(len(seq) - kmer + 1):
        mers.append(seq[i : i + kmer])
    mers = " ".join(mers)
    return mers


def match(seq_db: pd.DataFrame, TE_data: pd.DataFrame):
    seq_db_ids = seq_db["trans_id"].values
    TE_data_ids = TE_data["ensembl_tx_id"].values
    matched_id = set(seq_db_ids) & set(TE_data_ids)

    seq_db = seq_db.set_index("trans_id").loc[matched_id]
    te_value = TE_data.set_index("ensembl_tx_id").loc[matched_id]["te"]

    seq_db["te"] = te_value

    return seq_db


def mernize(matched_db):
    matched_db["fiveprime"] = list(map(seq2mer, matched_db["fiveprime"].values))
    matched_db["threeprime"] = list(map(seq2mer, matched_db["threeprime"].values))
    matched_db["cds"] = list(map(seq2mer, matched_db["cds"].values))
    return matched_db


def restrict_length(
    seq_db,
    five_min=50,
    five_max=500,
    cds_min=30,
    cds_max=2500,
    three_min=50,
    three_max=500,
):
    restricted_seq = seq_db[
        (seq_db["five_length"] > five_min)
        & (seq_db["five_length"] < five_max)
        & (seq_db["cds_length"] > cds_min)
        & (seq_db["cds_length"] < cds_max)
        & (seq_db["three_length"] > three_min)
        & (seq_db["three_length"] < three_max)
    ]

    return restricted_seq


def cut_higer_te(matched_df, threth=1000):
    matched_df = matched_df.sort_values("te", ascending=False).iloc[threth:]
    return matched_df


if __name__ == "__main__":
    args = _argparse()

    seq_db = pd.read_csv(args.sequence_db, index_col=0)
    TE_data = pd.read_table(args.te_data, sep=" ")
    TE_data = TE_data[TE_data.isnull().sum(axis=1) == 0]

    restricted_seq_db = restrict_length(seq_db)

    matched_df = match(restricted_seq_db, TE_data)
    matched_df = mernize(matched_df)
    # matched_df = cut_higer_te(matched_df)
    matched_df["te"] = np.log(matched_df["te"].values)  # log convertion
    matched_df.to_csv(os.path.join("../data/", args.save))
