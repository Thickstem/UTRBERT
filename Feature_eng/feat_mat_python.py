import os
import pandas as pd
import argparse
import gzip
from tqdm import tqdm


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--file_prefix")
    opt = args.parse_args()
    return opt


def main(opt):
    feat = pd.read_table(
        f"{opt.file_prefix}.txt.gz", header=None
    )  # It can read gunzip file directory
    feat.columns = ["tx_no", "feat_id", "val"]
    col = pd.read_table(f"{opt.file_prefix}.colname", header=None)
    row = pd.read_table(f"{opt.file_prefix}.rowname", header=None)
    before_dot = lambda id: id.split(".")[0]
    row["tx_id"] = list(map(before_dot, row.iloc[:, 1]))

    sparse_mat = pd.DataFrame(None, index=row["tx_id"], columns=col.iloc[:, 0])

    for id in tqdm(row.iloc[:, 0]):
        tx_df = feat[feat["tx_no"] == id]
        sparse_mat.iloc[id, tx_df["feat_id"].values] = tx_df["val"].values

    sparse_mat.to_csv(f"{opt.file_prefix}_final.csv")


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
