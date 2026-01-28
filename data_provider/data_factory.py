import copy
import re

import numpy as np
import torch

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Physio, Dataset_PEMS, Dataset_Epilepsy
from data_provider.uea import collate_fn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Electricity': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Weather': Dataset_Custom,
    'ECL': Dataset_Custom,
    'ILI': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'HAR': Dataset_Physio,
    'EEG': Dataset_Physio,
    'PEMS03': Dataset_PEMS,
    'PEMS04': Dataset_PEMS,
    'PEMS07': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'Epilepsy': Dataset_Epilepsy,
}

_SPLIT_PATTERN = re.compile(r"[,+]")


def _split_arg(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and _SPLIT_PATTERN.search(value):
        return [v.strip() for v in _SPLIT_PATTERN.split(value) if v.strip()]
    return [value]


class ChannelSplitDataset(Dataset):
    def __init__(self, base_dataset, drop_mark=False):
        self.base_dataset = base_dataset
        self.drop_mark = drop_mark
        self._n_channels = self._infer_channels()

    def _infer_channels(self):
        if len(self.base_dataset) == 0:
            return 1
        seq_x, _, _, _ = self.base_dataset[0]
        if torch.is_tensor(seq_x):
            shape = seq_x.shape
        else:
            shape = np.asarray(seq_x).shape
        if len(shape) <= 1:
            return 1
        return shape[-1]

    def _select_channel(self, arr, channel_idx):
        if torch.is_tensor(arr):
            if arr.ndim <= 1:
                return arr.unsqueeze(-1)
            return arr[..., channel_idx:channel_idx + 1]
        arr = np.asarray(arr)
        if arr.ndim <= 1:
            return arr[:, None]
        return arr[..., channel_idx:channel_idx + 1]

    def _zero_mark(self, length, ref):
        return torch.zeros((length, 1), dtype=torch.float32)

    def _to_tensor(self, arr, dtype=torch.float32):
        if torch.is_tensor(arr):
            return arr.to(dtype=dtype)
        return torch.tensor(arr, dtype=dtype)

    def __len__(self):
        return len(self.base_dataset) * self._n_channels

    def __getitem__(self, index):
        base_idx = index // self._n_channels
        channel_idx = index % self._n_channels
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base_dataset[base_idx]
        seq_x = self._select_channel(seq_x, channel_idx)
        seq_y = self._select_channel(seq_y, channel_idx)
        if self.drop_mark:
            seq_x_mark = self._zero_mark(len(seq_x), seq_x_mark)
            seq_y_mark = self._zero_mark(len(seq_y), seq_y_mark)
        seq_x = self._to_tensor(seq_x, dtype=torch.float32)
        seq_y = self._to_tensor(seq_y, dtype=torch.float32)
        seq_x_mark = self._to_tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = self._to_tensor(seq_y_mark, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark


def data_provider(args, flag):
    data_names = _split_arg(args.data)
    is_multi_dataset = len(data_names) > 1

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = True
        if args.downstream_task == 'anomaly_detection' or args.downstream_task == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if is_multi_dataset:
        if args.downstream_task in ('anomaly_detection', 'classification'):
            raise ValueError("Multi-dataset loading is only supported for forecasting tasks.")

        data_paths = _split_arg(args.data_path)
        root_paths = _split_arg(args.root_path)
        freqs = _split_arg(args.freq)

        if len(data_paths) not in (1, len(data_names)):
            raise ValueError(
                "Expected --data_path to be a single path or match --data entries ({} vs {}).".format(
                    len(data_paths), len(data_names)
                )
            )
        if len(root_paths) not in (1, len(data_names)):
            raise ValueError(
                "Expected --root_path to be a single path or match --data entries ({} vs {}).".format(
                    len(root_paths), len(data_names)
                )
            )
        if len(freqs) not in (1, len(data_names)):
            raise ValueError(
                "Expected --freq to be a single value or match --data entries ({} vs {}).".format(
                    len(freqs), len(data_names)
                )
            )

        combined_datasets = []
        for idx, data_name in enumerate(data_names):
            if data_name not in data_dict:
                raise ValueError("Unknown dataset type: {}".format(data_name))
            local_args = copy.copy(args)
            local_args.data = data_name
            local_args.data_path = data_paths[idx] if len(data_paths) > 1 else data_paths[0]
            local_args.root_path = root_paths[idx] if len(root_paths) > 1 else root_paths[0]
            local_args.freq = freqs[idx] if len(freqs) > 1 else freqs[0]

            Data = data_dict[local_args.data]
            data_set = Data(
                root_path=local_args.root_path,
                data_path=local_args.data_path,
                flag=flag,
                size=[local_args.input_len, local_args.label_len, local_args.pred_len],
                features=local_args.features,
                target=local_args.target,
                timeenc=timeenc,
                freq=local_args.freq,
                seasonal_patterns=local_args.seasonal_patterns
            )

            data_set = ChannelSplitDataset(data_set, drop_mark=True)
            combined_datasets.append(data_set)

        if args.data == 'm4' or 'm4' in data_names:
            drop_last = False

        data_set = ConcatDataset(combined_datasets)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        print(flag, len(data_set), len(data_loader))
        return data_set, data_loader

    Data = data_dict[args.data]

    if args.downstream_task == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.input_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
    elif args.downstream_task == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.input_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        print(flag, len(data_set), len(data_loader))
        return data_set, data_loader
