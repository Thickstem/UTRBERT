import os
import sys
import argparse
from utils import Seq2Feature
from multiprocessing import Pool, cpu_count
import gzip
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--df_path")
    args.add_argument("--out_dir")
    args.add_argument("--save")
    args.add_argument("--cds_len", type=int, default=0)
    args.add_argument("--energy", action="store_true")
    opt = args.parse_args()
    return opt


def single_test():
    featureDefineFn = sys.argv[1]
    seq = sys.argv[2]
    seq = Seq(seq)

    feature_id = dict()
    for line in open(featureDefineFn):
        comps = line.strip().split()
        feature_id[comps[1]] = int(comps[0])

    featList = Seq2Feature(seq)

    FeatureVec = [0] * len(feature_id)
    for item in featList:
        i = feature_id[item[0]]
        val = item[1]
        FeatureVec[i] = val

    print(" ".join(map(str, FeatureVec)))


def build_seq_list(opt, cds_length=0):
    df = pd.read_csv(opt.df_path, index_col=0)  # index = tx_id
    tx_ids = df["trans_id"].values
    # tx_ids = df.index.values

    if cds_length == 0:
        seqs = df["fiveprime"].values
    else:
        seqs = []
        for five, cds in zip(df["fiveprime"], df["cds"]):
            seq = five + cds[:cds_length]
            seqs.append(seq)

    bio_seqs = list(map(Seq, seqs))

    return tx_ids, bio_seqs
    return tx_ids, bio_seqs


def multi(opt):
    cds_length = opt.cds_len
    txIDlist, seq_list = build_seq_list(opt, cds_length)
    print(txIDlist)

    outputFasta = os.path.join(opt.out_dir, "input_sequence.fa")
    outf = open(outputFasta, "w")
    for tx_id, seq in zip(txIDlist, seq_list):
        outf.write(">" + tx_id + "\n")
        outf.write(str(seq) + "\n")

    converter = Seq2Feature(cds_length)
    pool = Pool(cpu_count())
    if opt.energy == True:
        featList = pool.map(converter.run_with_energy, seq_list)
    else:
        featList = pool.map(converter.run, seq_list)

    ####### Outputs

    outf2 = gzip.open(os.path.join(opt.out_dir, f"{opt.save}.txt.gz"), "wt")
    feat2ID = dict()
    featid = -1
    for i in range(len(txIDlist)):
        for featItem in featList[i]:
            featname = featItem[0]
            featVal = featItem[1]
            if featname not in feat2ID:
                featid += 1
                feat2ID[featname] = featid
                fid = featid
            else:
                fid = feat2ID[featname]
            outstr = str(i) + "\t" + str(fid) + "\t" + str(featVal)
            outf2.write(outstr + "\n")

    outf2.close()

    outf3 = open(os.path.join(opt.out_dir, f"{opt.save}.rowname"), "w")
    for i in range(len(txIDlist)):
        outf3.write(str(i) + "\t" + txIDlist[i] + "\n")

    outf3.close()

    outf4 = open(os.path.join(opt.out_dir, f"{opt.save}.colname"), "w")
    sorted_items = sorted(feat2ID.items(), key=lambda x: x[1])
    for a in sorted_items:
        featname = a[0]
        fid = a[1]
        outf4.write(str(fid) + "\t" + featname + "\n")
    outf4.close()


if __name__ == "__main__":
    opt = _argparse()
    os.makedirs(opt.out_dir, exist_ok=True)
    multi(opt)
