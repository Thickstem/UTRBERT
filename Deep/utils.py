import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def metrics(preds: np.ndarray, labels: np.ndarray):
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))

    return {"r2": r2, "mae": mae, "rmse": rmse}
