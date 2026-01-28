import numpy as np

from cps_robustness.dataset import TSDataset


def _unpack_item(item):
    if isinstance(item, (list, tuple)) and len(item) == 3:
        return item
    return item[0], item[1], None


def _repack_item(x, y, t_mark):
    if t_mark is None:
        return x, y
    return x, y, t_mark


class DriftDataset(TSDataset):
    """Add a constant offset to a random feature of the data."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.offset = self.set_params(severity)

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(
            int(self.n_features * prct_affected_sensors), len(self.continuous_features)
        )
        affected_sensors = self.rng.choice(
            self.continuous_features, n_affected_sensors, replace=False
        )

        min_offset = 0.0
        max_offset = 1.0
        offset = min_offset + severity * (max_offset - min_offset)

        return affected_sensors, offset

    def __getitem__(self, index):
        x, y, t_mark = _unpack_item(super().__getitem__(index))
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            x[:, sensor_idx] += self.offset
        return _repack_item(x, y, t_mark)


class DyingSignalDataset(TSDataset):
    """Multiply a random feature of the data with a constant factor."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors

        self.affected_sensors, self.factor = self.set_params(severity)

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(
            int(self.n_features * prct_affected_sensors), len(self.continuous_features)
        )
        affected_sensors = self.rng.choice(
            self.continuous_features, n_affected_sensors, replace=False
        )

        min_factor = 1.0
        max_factor = 0.0
        factor = min_factor + severity * (max_factor - min_factor)

        return affected_sensors, factor

    def __getitem__(self, index):
        x, y, t_mark = _unpack_item(super().__getitem__(index))
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            x[:, sensor_idx] *= self.factor
        return _repack_item(x, y, t_mark)


class NoiseDataset(TSDataset):
    """Add Gaussian noise to the data."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.sd = self.set_params(severity)
        self.noise = self._create_noise()

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(
            int(self.n_features * prct_affected_sensors), len(self.continuous_features)
        )
        affected_sensors = self.rng.choice(
            self.continuous_features, n_affected_sensors, replace=False
        )

        min_sd = 0.0
        max_sd = 1.0
        sd = min_sd + severity * (max_sd - min_sd)

        return affected_sensors, sd

    def _create_noise(self):
        full_noise = self.rng.normal(0, self.sd, (self.n_samples, self.input_len, self.n_features))
        noise = np.zeros((self.n_samples, self.input_len, self.n_features))
        for i in self.affected_sensors:
            idx = self.df.columns.get_loc(i)
            noise[:, :, idx] = full_noise[:, :, idx]
        return noise.astype(np.float32)

    def __getitem__(self, index):
        x, y, t_mark = _unpack_item(super().__getitem__(index))
        x_with_noise = x + self.noise[index]
        return _repack_item(x_with_noise, y, t_mark)


class FlatSensorDataset(TSDataset):
    """Set a random sensor to the last value for a random duration."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = min(target_prct_affected_sensors * 5, 1)
        self.affected_sensors, self.flat_duration = self.set_params(severity)
        self.flat_start_pos = self.rng.integers(
            1, self.input_len - self.flat_duration + 2, size=(self.n_samples, self.n_features)
        )

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = int(self.n_features * prct_affected_sensors)
        affected_sensors = self.rng.choice(self.feature_names, n_affected_sensors, replace=False)

        min_flat_duration = 1
        max_flat_duration = self.input_len
        flat_duration = int(min_flat_duration + severity * (max_flat_duration - min_flat_duration))

        return affected_sensors, flat_duration

    def __getitem__(self, index):
        x, y, t_mark = _unpack_item(super().__getitem__(index))
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            start_pos = self.flat_start_pos[index, sensor_idx]
            end_pos = start_pos + self.flat_duration
            last_valid_value = x[start_pos - 1, sensor_idx]
            x[start_pos:end_pos, sensor_idx] = last_valid_value
        return _repack_item(x, y, t_mark)


class MissingDataDataset(TSDataset):
    """Remove a random time window from the data."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=1.0, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.affected_sensors, self.missing_duration = self.set_params(severity)
        self.missing_start_pos = self.rng.integers(
            0, self.input_len - self.missing_duration, size=(self.n_samples)
        )

    def set_params(self, severity):
        affected_sensors = self.feature_names.values

        min_missing_duration = 1
        max_missing_duration = int(self.input_len * 0.5)
        missing_duration = min_missing_duration + int(
            severity * (max_missing_duration - min_missing_duration)
        )

        return affected_sensors, missing_duration

    def __getitem__(self, index):
        start_idx = self.sample_idxs[index]
        max_extra = len(self.df) - start_idx - self.input_len - self.target_len
        if max_extra <= 0:
            x, y, t_mark = _unpack_item(super().__getitem__(index))
            return _repack_item(x, y, t_mark)
        missing_duration = min(self.missing_duration, max_extra)
        if missing_duration <= 0:
            x, y, t_mark = _unpack_item(super().__getitem__(index))
            return _repack_item(x, y, t_mark)

        end_idx = start_idx + self.input_len + self.target_len + missing_duration
        df_ = self.df.iloc[start_idx:end_idx]

        missing_start = self.missing_start_pos[index]
        max_start = max(0, self.input_len - missing_duration)
        if missing_start > max_start:
            missing_start = max_start
        missing_end = missing_start + missing_duration
        df_ = df_.drop(df_.index[missing_start:missing_end])

        x = df_.iloc[: self.input_len].to_numpy().astype(np.float32)
        y = df_.iloc[self.input_len :].to_numpy().astype(np.float32)

        t_mark = None
        if getattr(self, "return_time_features", False) and getattr(
            self, "time_features_arr", None
        ) is not None:
            tf = self.time_features_arr[start_idx:end_idx]
            tf = np.delete(tf, np.s_[missing_start:missing_end], axis=0)
            t_mark = tf[: self.input_len].astype(np.float32)

        return _repack_item(x, y, t_mark)


class OutlierDataset(TSDataset):
    """Add an outlier to a random sensor of the data."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.hickup_value = self.set_params(severity)
        self.fault_mask = self._create_fault_mask()

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(
            int(self.n_features * prct_affected_sensors), len(self.continuous_features)
        )
        affected_sensors = self.rng.choice(
            self.continuous_features, n_affected_sensors, replace=False
        )

        min_hickup_value = 1
        max_hickup_value = 25
        hickup_value = min_hickup_value + severity * (max_hickup_value - min_hickup_value)

        return affected_sensors, hickup_value

    def _create_fault_mask(self):
        fault_mask = np.zeros((self.n_samples, self.input_len, self.n_features), dtype=np.float32)
        for sample in range(self.n_samples):
            for sensor in self.affected_sensors:
                sensor_idx = self.df.columns.get_loc(sensor)
                hickup_postion = self.rng.integers(1, self.input_len)
                fault_mask[sample, hickup_postion, sensor_idx] = self.hickup_value
        return fault_mask

    def __getitem__(self, index):
        x, y, t_mark = _unpack_item(super().__getitem__(index))
        x_with_fault = x + self.fault_mask[index]
        return _repack_item(x_with_fault, y, t_mark)


class FasterSamplingDataset(TSDataset):
    """Irregularly sample the data by warping the time axis of the input sequence."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = min(target_prct_affected_sensors * 5, 1)
        self.affected_sensors, self.warp_factor, self.warp_duration = self.set_params(severity)
        self.warp_start_pos = self.rng.integers(
            0, int(self.input_len * 0.5), size=(self.n_samples)
        )

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = int(self.n_features * prct_affected_sensors)
        affected_sensors = self.rng.choice(self.feature_names, n_affected_sensors, replace=False)

        min_warp_factor = 1.0
        max_warp_factor = 5.0
        warp_factor = min_warp_factor + severity * (max_warp_factor - min_warp_factor)

        warp_duration = int(self.input_len * 0.5)

        return affected_sensors, warp_factor, warp_duration

    def __getitem__(self, index):
        x, y, t_mark = _unpack_item(super().__getitem__(index))

        original_time_index = np.arange(self.input_len)
        irreg_time = np.full(self.warp_duration, self.warp_factor)
        irreg_time_index = np.cumsum(irreg_time) + self.warp_start_pos[index] - 1

        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            x[
                self.warp_start_pos[index] : (
                    self.warp_start_pos[index] + self.warp_duration
                ),
                sensor_idx,
            ] = np.interp(irreg_time_index, original_time_index, x[:, sensor_idx])

        return _repack_item(x, y, t_mark)


class SlowerSamplingDataset(TSDataset):
    """Irregularly sample the data by warping the time axis of the input sequence."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = min(target_prct_affected_sensors * 5, 1)
        self.affected_sensors, self.warp_factor, self.warp_duration = self.set_params(severity)
        self.warp_start_pos = self.rng.integers(
            0, int(self.input_len * 0.5), size=(self.n_samples)
        )

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = int(self.n_features * prct_affected_sensors)
        affected_sensors = self.rng.choice(self.feature_names, n_affected_sensors, replace=False)

        min_warp_factor = 1.0
        max_warp_factor = 0.0
        warp_factor = min_warp_factor + severity * (max_warp_factor - min_warp_factor)

        warp_duration = int(self.input_len * 0.5)

        return affected_sensors, warp_factor, warp_duration

    def __getitem__(self, index):
        x, y, t_mark = _unpack_item(super().__getitem__(index))

        original_time_index = np.arange(self.input_len)
        irreg_time = np.full(self.warp_duration, self.warp_factor)
        irreg_time_index = np.cumsum(irreg_time) + self.warp_start_pos[index] - 1

        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            x[
                self.warp_start_pos[index] : (
                    self.warp_start_pos[index] + self.warp_duration
                ),
                sensor_idx,
            ] = np.interp(irreg_time_index, original_time_index, x[:, sensor_idx])

        return _repack_item(x, y, t_mark)


class WrongDiscreteValueDataset(TSDataset):
    """A discrete sensor or actuator shows a wrong value."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.wrong_duration = self.set_params(severity)
        self.wrong_start_pos = self.rng.integers(
            1, self.input_len - self.wrong_duration + 2, size=(self.n_samples, self.n_features)
        )

    def set_params(self, severity):
        if len(self.discrete_features) == 0:
            raise ValueError("No discrete features available.")
        min_prct_affected_sensors = 1 / self.n_features + 1e-9
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(
            int(self.n_features * prct_affected_sensors), len(self.discrete_features)
        )
        affected_sensors = self.rng.choice(self.discrete_features, n_affected_sensors, replace=False)

        min_wrong_duration = 1
        max_wrong_duration = int(self.input_len / 10)
        wrong_duration = int(min_wrong_duration + severity * (max_wrong_duration - min_wrong_duration))

        return affected_sensors, wrong_duration

    def __getitem__(self, index):
        x, y, t_mark = _unpack_item(super().__getitem__(index))
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            start_pos = self.wrong_start_pos[index, sensor_idx]
            end_pos = start_pos + self.wrong_duration
            x[start_pos:end_pos, sensor_idx] = 2
        return _repack_item(x, y, t_mark)


class OscillatingSensorDataset(TSDataset):
    """A discrete sensor or actuator oscillates between two values."""

    def __init__(self, severity=1.0, target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.osc_duration = self.set_params(severity)
        self.osc_start_pos = self.rng.integers(
            1, self.input_len - self.osc_duration + 1, size=(self.n_samples, self.n_features)
        )

    def set_params(self, severity):
        if len(self.discrete_features) == 0:
            raise ValueError("No discrete features available.")
        min_prct_affected_sensors = 1 / self.n_features + 1e-9
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(
            int(self.n_features * prct_affected_sensors), len(self.discrete_features)
        )
        affected_sensors = self.rng.choice(self.discrete_features, n_affected_sensors, replace=False)

        min_osc_duration = 1
        max_osc_duration = int((self.input_len - 1) / 10)
        osc_duration = int(min_osc_duration + severity * (max_osc_duration - min_osc_duration))

        return affected_sensors, osc_duration

    def __getitem__(self, index):
        x, y, t_mark = _unpack_item(super().__getitem__(index))
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            start_pos = self.osc_start_pos[index, sensor_idx]
            end_pos = start_pos + self.osc_duration

            last_value = x[start_pos - 1, sensor_idx]
            if self.df[sensor].nunique() == 1:
                wrong_value = 1
            else:
                unique_values = self.df[sensor].unique().astype(np.float32)
                filtered_values = unique_values[unique_values != last_value]
                wrong_value = self.rng.choice(filtered_values)
            oscillating_values = self.rng.choice(
                [last_value, wrong_value], size=self.osc_duration, replace=True
            )
            x[start_pos:end_pos, sensor_idx] = oscillating_values

        return _repack_item(x, y, t_mark)
