import numpy as np
import itertools
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from Bio import SeqIO
import Bio.SeqUtils.CodonUsage
import subprocess
from multiprocessing import Pool
import gzip
from Bio.Seq import Seq


def metrics(preds: np.ndarray, labels: np.ndarray):
    print(f"preds:{preds}")
    print(f"label:{labels}")
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))

    return {"r2": r2, "mae": mae, "rmse": rmse}


def onehot_encode(seq):
    onehot_seq = []
    for s in seq.upper():
        if s == "A":
            onehot = [1, 0, 0, 0]
        elif s == "T":
            onehot = [0, 1, 0, 0]
        elif s == "G":
            onehot = [0, 0, 1, 0]
        elif s == "C":
            onehot = [0, 0, 0, 1]
        else:
            onehot = [0, 0, 0, 0]

        onehot_seq.append(onehot)

    return np.array(onehot_seq).T


class Kmer_count:
    def __init__(self, ks=[3], base_type="dna"):
        if base_type == "dna":
            base = ["A", "T", "G", "C"]
        elif base_type == "rna":
            base = ["A", "U", "G", "C"]
        else:
            raise TypeError

        self.kmer_dict = {}
        for k in ks:
            kmers = list(itertools.product(base, repeat=k))
            for kmer in kmers:
                str_kmer = "".join(b for b in kmer)
                self.kmer_dict[str_kmer] = 0

    def calc(self, seq, k=3, freq=True):
        count = 0
        tmp_dict = self.kmer_dict.copy()
        for st in range(len(seq) - k):
            kmer = seq[st : (st + k)]
            tmp_dict[kmer] += 1
            count += 1

        if freq:
            prob = np.array(list(tmp_dict.values())) / count
            return list(prob)
        else:
            return count


class Seq_n_padding:
    def __init__(
        self,
        regions=["fiveprime", "cds", "threeprime"],
        five_max: int = 500,
        cds_max: int = 2500,
        three_max: int = 500,
    ):
        self.region_dic = {}
        for reg in regions:
            if reg == "fiveprime":
                self.region_dic[reg] = five_max
            elif reg == "cds":
                self.region_dic[reg] = cds_max
            elif reg == "threeprime":
                self.region_dic[reg] = three_max
        print(f"reg_dict:{self.region_dic}")

    def padding(self, seqs):
        for i, reg in enumerate(self.region_dic.keys()):
            pad_len = self.region_dic[reg] - len(seqs[i])
            seqs[i] = seqs[i] + "N" * pad_len

        pad_seq = "".join(seqs)
        return pad_seq


def seq_n_padding(
    seqs: str,
    regions=["fiveprime", "cds", "threeprime"],
    five_max: int = 500,
    cds_max: int = 2500,
    three_max: int = 500,
):
    """

    Args:
        seq (str): [5'UTR,CDS,3'UTR]
        five_max (int, optional): _description_. Defaults to 500.
        cds_max (int, optional): _description_. Defaults to 2500.
        three_max (int, optional): _description_. Defaults to 500.
    """

    five_pad_len = five_max - len(seqs[0])
    cds_pad_len = cds_max - len(seqs[1])
    three_pad_len = three_max - len(seqs[2])

    seqs[0] = seqs[0] + "N" * five_pad_len
    seqs[1] = seqs[1] + "N" * cds_pad_len
    seqs[2] = seqs[2] + "N" * three_pad_len

    # print(len(seqs[0]), len(seqs[1]), len(seqs[2]))
    pad_seq = "".join(seqs)

    return pad_seq
