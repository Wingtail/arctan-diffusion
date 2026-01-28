import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_forecast_npy(name, root_path="datasets", univar=False):
    data_path = os.path.join(root_path, f"{name}.npy")
    data = np.load(data_path)
    if univar:
        data = data[: -1:]

    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt_index):
    dt = pd.DatetimeIndex(dt_index)
    week = dt.isocalendar().week.to_numpy()
    return np.stack(
        [
            dt.minute.to_numpy(),
            dt.hour.to_numpy(),
            dt.dayofweek.to_numpy(),
            dt.day.to_numpy(),
            dt.dayofyear.to_numpy(),
            dt.month.to_numpy(),
            week,
        ],
        axis=1,
    ).astype(np.float32)


def _read_csv_with_date(path):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def _select_univar(df, name_key):
    cols = list(df.columns)
    if name_key in ("etth1", "etth2", "ettm1", "ettm2") and "OT" in cols:
        return df[["OT"]]
    if name_key in ("electricity", "ecl") and "MT_001" in cols:
        return df[["MT_001"]]
    if name_key in ("wth", "weather") and "WetBulbCelsius" in cols:
        return df[["WetBulbCelsius"]]
    if "OT" in cols:
        return df[["OT"]]
    return df.iloc[:, -1:]


def load_forecast_csv(name, root_path="datasets", data_path=None, univar=False):
    name_key = name.lower()
    if data_path is None:
        data_path = f"{name}.csv"
    csv_path = os.path.join(root_path, data_path)
    df = _read_csv_with_date(csv_path)

    if isinstance(df.index, pd.DatetimeIndex):
        dt_embed = _get_time_features(df.index)
        n_covariate_cols = dt_embed.shape[-1]
    else:
        dt_embed = None
        n_covariate_cols = 0

    if univar:
        df = _select_univar(df, name_key)

    data = df.to_numpy()
    if name_key in ("etth1", "etth2"):
        train_slice = slice(None, 12 * 30 * 24)
        valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    elif name_key in ("ettm1", "ettm2"):
        train_slice = slice(None, 12 * 30 * 24 * 4)
        valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    elif name_key.startswith("m5"):
        train_slice = slice(None, int(0.8 * (1913 + 28)))
        valid_slice = slice(int(0.8 * (1913 + 28)), 1913 + 28)
        test_slice = slice(1913 + 28 - 1, 1913 + 2 * 28)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name_key in ("electricity", "ecl") or name_key.startswith("m5"):
        data = np.expand_dims(data.T, -1)
    else:
        data = np.expand_dims(data, 0)

    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate(
            [np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1
        )

    if name_key in ("etth1", "etth2", "electricity", "ecl", "wth", "weather"):
        pred_lens = [24, 48, 168, 336, 720]
    elif name_key.startswith("m5"):
        pred_lens = [28]
    else:
        pred_lens = [24, 48, 96, 288, 672]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols
