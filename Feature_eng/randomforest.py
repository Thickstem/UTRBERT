import os
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--te_df",
        required=True,
        default="../df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt",
    )
    parser.add_argument("--feature", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--riboseq_thresh", type=float, default=0.1)
    parser.add_argument("--rnaseq_thresh", type=float, default=5)

    opt = parser.parse_args()
    return opt


def metrics(preds: np.ndarray, labels: np.ndarray):
    print(f"preds:{preds}")
    print(f"label:{labels}")
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    spearman, _ = spearmanr(labels, preds)  # corr, p_val
    pearson, _ = pearsonr(labels, preds)  # corr, p_val

    return {
        "r2": r2,
        "spearman": spearman,
        "pearson": pearson,
        "mae": mae,
        "rmse": rmse,
    }


def match(feat: pd.DataFrame, TE_data: pd.DataFrame):
    feat_ids = feat.index.values
    TE_data_ids = TE_data["ensembl_tx_id"].values
    matched_id = list(set(feat_ids) & set(TE_data_ids))

    feat = feat.loc[matched_id]
    te_value = TE_data.set_index("ensembl_tx_id").loc[matched_id]["te"]

    feat["te"] = np.log(te_value)

    return feat


def data_preparation(opt: argparse.Namespace):
    TE_data = pd.read_table(opt.te_df, sep=" ")
    TE_data = TE_data[TE_data.isnull().sum(axis=1) == 0]
    TE_data = TE_data[
        (TE_data["rpkm_riboseq"] > opt.riboseq_thresh)
        & (TE_data["rpkm_rnaseq"] > opt.rnaseq_thresh)
    ]

    feature_mat = pd.read_csv(opt.feature, index_col=0)  # sparse matrix
    feature_mat = feature_mat.fillna(0).astype(pd.SparseDtype("float64", 0))

    feat_te = match(feature_mat, TE_data)
    return feat_te


def main():
    opt = _parse_args()
    t1 = time.time()
    print("Data preparing...")
    data = data_preparation(opt)  # feat_te:[trans_id,feature+te]
    print(f"Data size:{data.shape}")
    print("")
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2
    )

    model = RandomForestRegressor()

    print(f"Model fitting...")
    model.fit(X_train, y_train)
    print(f"Predicting...")
    preds = model.predict(X_test)
    print(f"predicting time:{(time.time()-t1):.3f}")

    scores = metrics(preds, labels=y_test.values)

    with open(os.path.join(opt.save_dir, "RF_res.txt"), "w") as f:
        for k, v in scores.items():
            print(f"{k}:{v:.4f}")
            f.write(f"{k}:{v:.4f}\n")
    print(f"Elapsed time:{(time.time()-t1):.3f} s")


if __name__ == "__main__":
    main()
