import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from cps_robustness.dataset import TSDataset
from cps_robustness.disturbances import (
    DriftDataset,
    DyingSignalDataset,
    NoiseDataset,
    FlatSensorDataset,
    MissingDataDataset,
    FasterSamplingDataset,
    SlowerSamplingDataset,
    OutlierDataset,
    WrongDiscreteValueDataset,
    OscillatingSensorDataset,
)


class CPSRobustnessData:
    """Minimal data module for the CPS robustness benchmark (no Lightning dependency)."""

    def __init__(
        self,
        file_path,
        input_len=100,
        target_len=20,
        stride=1,
        n_train_samples=None,
        n_val_samples=None,
        n_test_samples=128,
        n_severity_levels=101,
        prct_affected_sensors=0.1,
        train_split=0.7,
        val_split=0.15,
        purged_fraction=0.01,
        batch_size=64,
        num_workers=0,
        seed=42,
        features="M",
        target=None,
        train_df=None,
        val_df=None,
        test_df=None,
        already_scaled=False,
        return_time_features=False,
        time_features_freq="h",
        time_features_mode="timeF",
    ):
        self.file_path = file_path
        self.input_len = input_len
        self.target_len = target_len
        self.stride = stride
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.n_severity_levels = n_severity_levels
        self.prct_affected_sensors = prct_affected_sensors
        self.train_split = train_split
        self.val_split = val_split
        self.purged_fraction = purged_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.features = features
        self.target = target
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.already_scaled = already_scaled
        self.return_time_features = return_time_features
        self.time_features_freq = time_features_freq
        self.time_features_mode = time_features_mode

        self.dataset_mapping = {
            "normal": TSDataset,
            "drift": DriftDataset,
            "dying_signal": DyingSignalDataset,
            "noise": NoiseDataset,
            "flat_sensor": FlatSensorDataset,
            "missing_data": MissingDataDataset,
            "faster_sampling": FasterSamplingDataset,
            "slower_sampling": SlowerSamplingDataset,
            "outlier": OutlierDataset,
            "wrong_discrete_value": WrongDiscreteValueDataset,
            "oscillating_sensor": OscillatingSensorDataset,
        }

        self.ds_train = None
        self.ds_val = None
        self.ds_test_dict = {}
        self.test_dataloader_mapping = pd.DataFrame(
            columns=["dataloader_idx", "scenario", "severity"]
        ).set_index("dataloader_idx")

        self.train_mean_vals = None
        self.train_sd_vals = None
        self.feature_names = None
        self.discrete_features = None

    def _get_split_indices(self, len_df):
        train_end_idx = int(len_df * self.train_split)
        val_start_idx = train_end_idx
        val_end_idx = int(len_df * (self.train_split + self.val_split))

        purged_val_start_idx = val_start_idx + int(len_df * self.purged_fraction)
        purged_test_start_idx = val_end_idx + int(len_df * self.purged_fraction)

        assert train_end_idx < len_df
        assert purged_val_start_idx < len_df
        assert val_end_idx < len_df
        assert purged_test_start_idx < len_df

        return train_end_idx, purged_val_start_idx, val_end_idx, purged_test_start_idx

    def _read_dataframe(self):
        if self.file_path.endswith(".parquet"):
            df_all = pd.read_parquet(self.file_path)
        elif self.file_path.endswith(".csv"):
            df_all = pd.read_csv(self.file_path)
            try:
                df_all.set_index(
                    pd.to_datetime(df_all.iloc[:, 0], format="%Y-%m-%d %H:%M:%S"),
                    inplace=True,
                )
                df_all.drop(df_all.columns[0], axis=1, inplace=True)
            except (ValueError, TypeError):
                pass
        else:
            raise ValueError("File format not supported.")

        if self.features == "S":
            if self.target is None:
                raise ValueError("features='S' requires --target.")
            if self.target not in df_all.columns:
                raise ValueError(f"Target column '{self.target}' not found in dataset.")
            df_all = df_all[[self.target]]
        elif self.features in ("M", "MS") and self.target in df_all.columns:
            cols = [col for col in df_all.columns if col != self.target] + [self.target]
            df_all = df_all[cols]

        self.feature_names = df_all.columns
        return df_all

    def setup(self):
        if self.train_df is not None and self.test_df is not None:
            df_train = self.train_df.copy()
            df_val = self.val_df.copy() if self.val_df is not None else None
            df_test = self.test_df.copy()

            self.feature_names = df_train.columns
            if not self.already_scaled:
                self.train_mean_vals = df_train.mean()
                self.train_sd_vals = df_train.std()
                self.train_sd_vals.replace(0, 1.0, inplace=True)
                df_train = (df_train - self.train_mean_vals) / self.train_sd_vals
                if df_val is not None:
                    df_val = (df_val - self.train_mean_vals) / self.train_sd_vals
                df_test = (df_test - self.train_mean_vals) / self.train_sd_vals
        else:
            df_all = self._read_dataframe()
            self.feature_names = df_all.columns

            train_end, purged_val_start, val_end, purged_test_start = self._get_split_indices(
                len(df_all)
            )
            df_train = df_all.iloc[:train_end, :]
            df_val = df_all.iloc[purged_val_start:val_end, :]
            df_test = df_all.iloc[purged_test_start:, :]
            del df_all

            self.train_mean_vals = df_train.mean()
            self.train_sd_vals = df_train.std()
            self.train_sd_vals.replace(0, 1.0, inplace=True)
            df_train = (df_train - self.train_mean_vals) / self.train_sd_vals
            df_val = (df_val - self.train_mean_vals) / self.train_sd_vals
            df_test = (df_test - self.train_mean_vals) / self.train_sd_vals

        self.ds_train = TSDataset(
            df=df_train,
            input_len=self.input_len,
            target_len=self.target_len,
            stride=self.stride,
            n_samples=self.n_train_samples,
            seed=self.seed,
            return_time_features=False,
            time_features_freq=self.time_features_freq,
            time_features_mode=self.time_features_mode,
        )
        self.discrete_features = self.ds_train.discrete_features
        del df_train

        if df_val is not None:
            self.ds_val = TSDataset(
                df=df_val,
                input_len=self.input_len,
                target_len=self.target_len,
                stride=self.stride,
                n_samples=self.n_val_samples,
                continuous_features=self.ds_train.continuous_features,
                discrete_features=self.ds_train.discrete_features,
                seed=self.seed,
                return_time_features=False,
                time_features_freq=self.time_features_freq,
                time_features_mode=self.time_features_mode,
            )
            del df_val

        dataset_mapping = dict(self.dataset_mapping)
        if len(self.ds_train.discrete_features) == 0:
            dataset_mapping.pop("wrong_discrete_value", None)
            dataset_mapping.pop("oscillating_sensor", None)
        self.dataset_mapping = dataset_mapping

        dataloader_idx = 0
        for scenario in self.dataset_mapping.keys():
            if scenario == "normal":
                self.test_dataloader_mapping.loc[dataloader_idx] = [scenario, np.nan]
                dataloader_idx += 1
                self.ds_test_dict[scenario] = [
                    self.dataset_mapping[scenario](
                        df=df_test,
                        input_len=self.input_len,
                        target_len=self.target_len,
                        stride=self.stride,
                        n_samples=self.n_test_samples,
                        continuous_features=self.ds_train.continuous_features,
                        discrete_features=self.ds_train.discrete_features,
                        seed=self.seed,
                        return_time_features=self.return_time_features,
                        time_features_freq=self.time_features_freq,
                        time_features_mode=self.time_features_mode,
                    )
                ]
            else:
                scenario_datasets = []
                for severity in np.linspace(0, 1, self.n_severity_levels):
                    self.test_dataloader_mapping.loc[dataloader_idx] = [scenario, severity]
                    dataloader_idx += 1
                    scenario_datasets.append(
                        self.dataset_mapping[scenario](
                            df=df_test,
                            severity=severity,
                            target_prct_affected_sensors=self.prct_affected_sensors,
                            input_len=self.input_len,
                            target_len=self.target_len,
                            stride=self.stride,
                            n_samples=self.n_test_samples,
                            continuous_features=self.ds_train.continuous_features,
                            discrete_features=self.ds_train.discrete_features,
                            seed=self.seed,
                            return_time_features=self.return_time_features,
                            time_features_freq=self.time_features_freq,
                            time_features_mode=self.time_features_mode,
                        )
                    )
                self.ds_test_dict[scenario] = scenario_datasets
        del df_test

    def build_test_loaders(self):
        if not self.ds_test_dict:
            raise RuntimeError("Call setup() before building test loaders.")

        test_dl_list = [
            DataLoader(
                self.ds_test_dict[scenario][severity_level],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=False,
            )
            for scenario in self.dataset_mapping.keys()
            for severity_level in range(len(self.ds_test_dict[scenario]))
        ]
        return test_dl_list, self.test_dataloader_mapping

    def get_target_index(self):
        if self.features == "S":
            return 0
        if self.features == "MS" and self.target in self.feature_names:
            return int(self.feature_names.get_loc(self.target))
        return None
