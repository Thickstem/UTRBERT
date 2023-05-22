from tqdm import tqdm
import hydra
import numpy as np
import pandas as pd

from utils import Kmer_count

from omegaconf import DictConfig


def match(cfg: DictConfig, seq_df: pd.DataFrame, te_df: pd.DataFrame) -> pd.DataFrame:
    te_df = te_df[te_df.isnull().sum(axis=1) == 0]
    te_df = te_df[
        (te_df["rpkm_riboseq"] > cfg.riboseq_thresh)
        & (te_df["rpkm_rnaseq"] > cfg.rnaseq_thresh)
    ]

    seq_ids = seq_df.index.values
    te_df_ids = te_df["ensembl_tx_id"].values
    matched_id = list(set(seq_ids) & set(te_df_ids))

    seq_df = seq_df.loc[matched_id]
    te_value = te_df.set_index("ensembl_tx_id").loc[matched_id]["te"]

    seq_df["te"] = np.log(te_value)

    return seq_df


def simple_padding(seq: str, max_seq_len: int) -> str:
    """padding input seq to max_seq_len

    Args:
        seq (str): input sequence
        max_seq_len (int): maximum padding length

    Returns:
        str: pad sequence
    """
    pad_len = max_seq_len - len(seq)
    pad_seq = seq + "N" * pad_len
    return pad_seq


def build_seq_df(
    seq_df: pd.DataFrame, cds_length=0, min_seq_len=30, max_seq_len=500
) -> pd.DataFrame:
    """building DF. concating,thresholding,Kmer_counting,padding sequences

    Args:
        seq_df (pd.DataFrame): raw sequence dataframe
        cds_length (int, optional): length of CDS including to input seq. Defaults to 0.
        min_seq_len (int, optional): minimum seq len to use as input. Defaults to 30.
        max_seq_len (int, optional): maximum seq len to use as input. Defaults to 500.

    Returns:
        pd.DataFrame: built df. only contains index and seq (1 column)
    """
    if cds_length == 0:
        seqs = seq_df["fiveprime"].values
    else:  # concat fiveprime and CDS head
        seqs = []
        for five, cds in zip(seq_df["fiveprime"], seq_df["cds"]):
            seq = five + cds[:cds_length]
            seqs.append(seq)

    kmer_counter = Kmer_count()

    id_list = []
    seq_list = []
    for id, seq in tqdm(zip(seq_df["trans_id"], seqs)):
        if (len(seq) >= min_seq_len) & (len(seq) <= max_seq_len):
            kmer_prob = kmer_counter.calc(seq)
            pad_seq = simple_padding(seq, max_seq_len)
            id_list.append(id)
            seq_feature = [pad_seq]
            seq_feature.extend(kmer_prob)
            seq_list.append(seq_feature)
    col_names = ["seq"]
    col_names.extend(kmer_counter.kmer_dict.keys())
    built_df = pd.DataFrame(seq_list, index=id_list, columns=col_names)
    return built_df


@hydra.main(config_path="/home/ksuga/UTRBERT/Deep/configs", config_name="cnn_data")
def main(cfg: DictConfig) -> None:
    seq_df = pd.read_csv(cfg.seq_df_path, index_col=0)
    te_df = pd.read_table(cfg.te_df_path, sep=" ")

    built_df = build_seq_df(seq_df)
    matched_df = match(cfg, built_df, te_df)

    matched_df.to_csv(cfg.output)


if __name__ == "__main__":
    main()
