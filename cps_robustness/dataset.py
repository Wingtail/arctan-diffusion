import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class TSDataset(Dataset):
    """Time Series Dataset.

    A sample consists of a (random) time window + consecutive time horizon.
    """

    def __init__(
        self,
        df=None,
        file_path=None,
        input_len=90,
        target_len=30,
        stride=1,
        n_samples=None,
        mean_vals=None,
        sd_vals=None,
        continuous_features=None,
        discrete_features=None,
        seed=42,
        return_time_features=False,
        time_features_freq="h",
        time_features_mode="timeF",
    ):
        super().__init__()
        if df is not None:
            self.df = df
        elif file_path is not None:
            if file_path.endswith(".parquet"):
                self.df = pd.read_parquet(file_path)
            elif file_path.endswith(".csv"):
                self.df = pd.read_csv(file_path)
            else:
                raise ValueError("File format not supported.")
            try:
                self.df.set_index(
                    pd.to_datetime(self.df.iloc[:, 0], format="%Y-%m-%d %H:%M:%S"),
                    inplace=True,
                )
                self.df.drop(self.df.columns[0], axis=1, inplace=True)
            except ValueError:
                pass
        else:
            raise ValueError("Either df or file_path must be given.")

        self.input_len = input_len
        self.target_len = target_len
        self.stride = stride
        self.return_time_features = return_time_features
        self.time_features_freq = time_features_freq
        self.time_features_mode = time_features_mode

        self.random_sampling = n_samples is not None
        self.n_samples = self.__len__() if n_samples is None else n_samples
        assert (
            self.n_samples <= self.__len__()
        ), "n_samples must be smaller than the number of possible samples."
        self.n_features = self.df.shape[1]
        self.feature_names = self.df.columns
        self.continuous_features, self.discrete_features = self.split_hybrid_data(
            continuous_features, discrete_features
        )

        # Rescale data if mean and sd values are given.
        self.mean_vals = mean_vals
        self.sd_vals = sd_vals
        if mean_vals is not None and sd_vals is not None:
            self.scale_data()

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.time_features_arr = None
        if self.return_time_features:
            if not isinstance(self.df.index, pd.DatetimeIndex):
                raise ValueError(
                    "return_time_features=True requires a datetime index on the dataframe."
                )
            dates = pd.to_datetime(self.df.index)
            if self.time_features_mode == "timeF":
                feats = time_features(dates, freq=self.time_features_freq).transpose(1, 0)
            else:
                df_stamp = pd.DataFrame({"date": dates})
                df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
                feats = df_stamp.drop(columns=["date"]).values
            self.time_features_arr = feats.astype(np.float32)

        self.sample_idxs = self._create_sample_indices()

    def split_hybrid_data(self, continuous_features=None, discrete_features=None):
        """Split the time series data features into continuous and discrete features."""
        continuous_threshold = 32
        continuous_features = (
            [feature for feature in self.df.columns if self.df[feature].nunique() > continuous_threshold]
            if continuous_features is None
            else continuous_features
        )
        discrete_features = (
            [feature for feature in self.df.columns if self.df[feature].nunique() <= continuous_threshold]
            if discrete_features is None
            else discrete_features
        )
        assert (
            len(continuous_features) + len(discrete_features) == self.n_features
        ), "All features must be assigned to either continuous or discrete features."
        return continuous_features, discrete_features

    def set_scaler_params(self, mean_vals=None, sd_vals=None):
        """Set the parameters for scaling the data."""
        self.mean_vals = mean_vals if mean_vals is not None else self.df.mean()
        self.sd_vals = sd_vals if sd_vals is not None else self.df.std()

    def scale_data(self):
        """Standardize the data."""
        assert (
            self.mean_vals is not None and self.sd_vals is not None
        ), "Mean and standard deviation values must be set first."
        self.sd_vals.replace(0, 1.0, inplace=True)
        self.df = (self.df - self.mean_vals) / self.sd_vals

    def inverse_scale_data(self, scaled_data):
        df_ = pd.DataFrame(scaled_data, columns=self.df.columns)
        return (df_ * self.sd_vals) + self.mean_vals

    def _create_sample_indices(self):
        """Create an array of indices for sampling."""
        max_start = self.df.shape[0] - self.input_len - self.target_len
        if max_start <= 0:
            raise ValueError(
                "Not enough data to create samples. "
                f"Need at least input_len+target_len={self.input_len + self.target_len} rows."
            )
        if self.random_sampling:
            sample_idxs = self.rng.integers(low=0, high=max_start, size=self.n_samples)
        else:
            max_n_samples = int(max_start / self.stride) + 1
            sample_idxs = np.arange(max_n_samples) * self.stride
        return sample_idxs

    def __len__(self):
        if self.random_sampling:
            return self.n_samples
        return int((self.df.shape[0] - self.input_len - self.target_len) / self.stride) + 1

    def __getitem__(self, index):
        start_idx = self.sample_idxs[index]
        end_idx = start_idx + self.input_len + self.target_len
        df_ = self.df.iloc[start_idx:end_idx]
        x = df_.iloc[: self.input_len].to_numpy().astype(np.float32)
        y = df_.iloc[self.input_len :].to_numpy().astype(np.float32)
        del df_
        if self.return_time_features:
            t_mark = self.time_features_arr[start_idx : start_idx + self.input_len]
            return x, y, t_mark
        return x, y
