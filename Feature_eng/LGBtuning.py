import os
import time
import json
import logging
import logging.config
import argparse

import pandas as pd
import numpy as np
import lightgbm as lgb
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
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument("--config", default="./scripts/conf.json")
    parser.add_argument("--res_file", default="results")

    opt = parser.parse_args()
    return opt


def read_conf_file(conf_file):
    with open(conf_file, "r", encoding="utf-8") as f:
        file = json.load(f)
        logging.config.dictConfig(file)


def get_logger(logger_="simpleDefault"):
    """Generating Logger"""
    return logging.getLogger(logger_)


def metrics(preds: np.ndarray, labels: np.ndarray):
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


def main(opt, logger):
    logger.debug(vars(opt))
    t1 = time.time()
    logger.debug("Data preparing...")
    data = data_preparation(opt)  # feat_te:[trans_id,feature+te]
    logger.debug(f"Data size:{data.shape}")

    if opt.cv == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=opt.test_size, random_state=0
        )

        if opt.model == "rf":
            model = RandomForestRegressor()
        elif opt.model == "lgb":
            model = lgb.LGBMRegressor()
        else:
            raise NameError()

        logger.debug(f"Model fitting...")
        model.fit(X_train, y_train)
        logger.debug(f"Predicting...")
        preds = model.predict(X_test)
        logger.debug(f"predicting time:{(time.time()-t1):.3f}")

        scores = metrics(preds, labels=y_test.values)

        with open(os.path.join(opt.save_dir, opt.res_file + ".txt"), "w") as f:
            for k, v in scores.items():
                logger.debug(f"{k}:{v:.4f}")
                f.write(f"{k}:{v:.4f}\n")
        logger.debug(f"Elapsed time:{(time.time()-t1):.3f} s")

    else:
        logger.debug(f"Iterate for {opt.cv} times")
        score_dict = {}
        with open(os.path.join(opt.save_dir, opt.res_file + ".txt"), "w") as res_f:
            for i in range(opt.cv):
                X_train, X_test, y_train, y_test = train_test_split(
                    data.iloc[:, :-1],
                    data.iloc[:, -1],
                    test_size=opt.test_size,
                    random_state=i,
                )
                """
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_raw, y_train_raw, test_size=0.2
                )
                """
                params = {
                    "objective": "regression",  # 最小化させるべき損失関数
                    "random_state": 42,  # 乱数シード
                    "boosting_type": "gbdt",  # boosting_type
                    "n_estimators": 10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
                }
                verbose_eval = 0
                logger.debug("model: LightGBM")
                model = lgb.LGBMRegressor()

                logger.debug(f"[{i+1}/{opt.cv}]:Model fitting... ")
                model.fit(
                    X_train,
                    y_train,
                    eval_metric="rmse",
                    eval_set=[(X_test, y_test)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=10, verbose=True),
                        lgb.log_evaluation(verbose_eval),
                    ],
                )
                logger.debug(f"[{i+1}/{opt.cv}]:Predicting...")
                preds = model.predict(X_test)
                logger.debug(
                    f"[{i+1}/{opt.cv}]:predicting time:{(time.time()-t1):.3f} s"
                )

                scores = metrics(preds, labels=y_test.values)

                for k, v in scores.items():
                    logger.debug(f"[{i+1}/{opt.cv}]: {k}:{v:.4f}")
                    res_f.write(f"[{i+1}/{opt.cv}]: {k}:{v:.4f}\n")

                    if k not in score_dict.keys():
                        score_dict[k] = [v]
                    else:
                        score_dict[k].append(v)

                res_f.write("\n")

            stats_dict = {"score": [], "mean": [], "var": []}

            for k, v in score_dict.items():
                stats_dict["score"].append(k)
                stats_dict["mean"].append(np.mean(v))
                stats_dict["var"].append(np.var(v))

            pd.DataFrame(stats_dict).to_csv(
                os.path.join(opt.save_dir, opt.res_file + ".csv"), index=False
            )

            logger.debug(f"Elapsed time:{(time.time()-t1):.3f} s")


if __name__ == "__main__":
    opt = _parse_args()
    read_conf_file(opt.config)
    logger = get_logger(logger_=os.path.basename(opt.feature.replace("_final.csv", "")))
    main(opt, logger)
