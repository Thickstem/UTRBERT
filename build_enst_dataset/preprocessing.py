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


def match(seq_db: pd.DataFrame, TE_data: pd.DataFrame):
    seq_db_ids = seq_db["trans_id"].values
    TE_data_ids = TE_data["ensembl_tx_id"].values
    matched_id = set(seq_db_ids) & set(TE_data_ids)

    seq_db_ids = seq_db_ids.set_index("trans_id").loc[matched_id]
    te_value = TE_data_ids.set_index("ensembl_tx_id").loc[matched_id]["te"]

    seq_db_ids["te"] = te_value

    return seq_db_ids


if __name__ == "__main__":
    args = _argparse()

    seq_db = pd.read_csv(args.sequence_db, index_col=0)
    TE_data = pd.read_csv(args.te_data, index_col=0)
    TE_data = TE_data[TE_data.isnull().sum(axis=1) == 0]

    matched_df = match(seq_db, TE_data)
    matched_df.to_csv(os.path.join("../data/", args.save))
