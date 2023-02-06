import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def metrics(preds: np.ndarray, labels: np.ndarray):
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

    return onehot_seq


def seq_n_padding(
    seqs: str, five_max: int = 500, cds_max: int = 2500, three_max: int = 500
):
    """

    Args:
        seq (str): [5'UTR,CDS,3'UTR]
        five_max (int, optional): _description_. Defaults to 500.
        cds_max (int, optional): _description_. Defaults to 2500.
        three_max (int, optional): _description_. Defaults to 500.
    """

    five_pad_len = five_max - seqs[0]
    cds_pad_len = cds_max - seqs[1]
    three_pad_len = three_max - seqs[2]

    seqs[0] = seqs[0] + "N" * five_pad_len
    seqs[1] = seqs[1] + "N" * cds_pad_len
    seqs[2] = seqs[2] + "N" * three_pad_len

    pad_seq = "".join(seqs)

    return pad_seq
