import os
import pandas as pd
import numpy as np

from torch_timeseries.core import TimeSeriesDataset, Freq, TimeseriesSubset
from torch.utils.data import ConcatDataset, DataLoader
from torch_timeseries.dataloader.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.scaler import Scaler


class CustomMultyCsv(TimeSeriesDataset):
    """
    Custom Multi-CSV dataset loader. Expects multiple CSV files in data/custom directory.
    Reads each CSV as an independent time series. Generates a combined dataset by
    concatenating all series for compatibility with SlidingWindowTS, and stores
    boundaries for potential custom loaders.

    Usage:
        Place one or more CSV files in data/custom/. Each CSV should have the first
        column as datetime and remaining columns as features. Filenames can be arbitrary.

    Attributes set:
        series_list: list of np.ndarray, each series data
        dates_list: list of pd.DataFrame, each series dates
        series_lengths: list of int, lengths of each series
        series_boundaries: list of int, cumulative end indices for each series
        data: np.ndarray, concatenated data of all series
        dates: pd.DataFrame, concatenated dates of all series
        num_series: int, number of CSV series loaded
        num_features: int, number of features per time step
        length: int, total number of time steps across all series
        freq: Freq, frequency inferred from the first series (others assumed same)
    """
    name: str = 'custom_multy'

    def download(self):
        # No download needed; data should exist in data/custom
        pass

    def _process(self):
        # No pre-processing step
        pass

    def _load(self) -> np.ndarray:
        # Load all CSV files in custom directory
        custom_dir = os.path.join(self.root, 'custom')
        if not os.path.isdir(custom_dir):
            raise FileNotFoundError(f"Custom CSV directory not found: {custom_dir}")
        csv_files = [f for f in os.listdir(custom_dir) if f.lower().endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {custom_dir}")

        series_list = []
        dates_list = []
        lengths = []

        for fname in sorted(csv_files):
            path = os.path.join(custom_dir, fname)
            df = pd.read_csv(path, parse_dates=[0])
            if df.shape[1] < 2:
                raise ValueError(
                    f"Each CSV must contain at least two columns (time + features), got {df.shape} in {fname}"
                )
            # Extract times and values
            times = df.iloc[:, 0]
            values = df.iloc[:, 1:].to_numpy()
            series_list.append(values)
            dates_list.append(pd.DataFrame({'date': times}))
            lengths.append(values.shape[0])

        # Store series attributes
        self.series_list = series_list
        self.dates_list = dates_list
        self.series_lengths = lengths
        # Compute cumulative boundaries
        boundaries = np.cumsum(lengths).tolist()
        self.series_boundaries = boundaries
        self.num_series = len(series_list)

        # For compatibility, concatenate series for SlidingWindowTS
        self.data = np.concatenate(series_list, axis=0)
        self.dates = pd.concat(dates_list, ignore_index=True)

        # Set dataset properties
        self.num_features = self.data.shape[1]
        self.length = self.data.shape[0]

        # Infer frequency from first series dates
        freq_str = pd.infer_freq(self.dates_list[0]['date'])
        if freq_str and freq_str[0].lower() in Freq._value2member_map_:
            self.freq = Freq(freq_str[0].lower())
        else:
            # Default to daily frequency
            self.freq = Freq.days

        return self.data


from torch.utils.data import ConcatDataset, DataLoader
from torch_timeseries.core import TimeseriesSubset
from torch_timeseries.dataloader.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.scaler import Scaler


class CustomMultyLoader:
    """
    Custom loader for multiple CSV time series. Splits each series independently
    into train/val/test and produces per-series sliding-window datasets,
    then concatenates them so windows never cross series.
    """
    def __init__(
        self,
        dataset: CustomMultyCsv,
        scaler: Scaler,
        time_enc: int = 0,
        window: int = 168,
        horizon: int = 3,
        steps: int = 2,
        scale_in_train: bool = True,
        shuffle_train: bool = True,
        freq=None,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        num_worker: int = 3,
        uniform_eval: bool = True,
        single_variate: bool = False,
    ):
        self.dataset = dataset
        self.scaler = scaler
        self.time_enc = time_enc
        self.window = window
        self.horizon = horizon
        self.steps = steps
        self.scale_in_train = scale_in_train
        self.shuffle_train = shuffle_train
        self.freq = freq or dataset.freq
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.num_worker = num_worker
        self.uniform_eval = uniform_eval
        self.single_variate = single_variate

        self._load()

    def _load(self):
        # per-series splitting
        boundaries = [0] + self.dataset.series_boundaries
        train_ranges, val_ranges, test_ranges = [], [], []
        for i in range(self.dataset.num_series):
            start = boundaries[i]
            end = boundaries[i+1]
            length = end - start
            train_size = int(self.train_ratio * length)
            test_size = int(self.test_ratio * length)
            val_size = length - train_size - test_size

            # training indices
            train_idx = list(range(start, start + train_size))

            # validation indices
            if self.uniform_eval:
                offset = train_size - self.window - self.horizon + 1
                val_start = start + max(offset, 0)
                val_idx = list(range(val_start, start + train_size + val_size))
            else:
                val_idx = list(range(start + train_size, start + train_size + val_size))

            # test indices
            if self.uniform_eval:
                offset_t = length - test_size - self.window - self.horizon + 1
                test_start = start + max(offset_t, 0)
                test_idx = list(range(test_start, end))
            else:
                test_idx = list(range(end - test_size, end))

            train_ranges.append(train_idx)
            val_ranges.append(val_idx)
            test_ranges.append(test_idx)

        # fit scaler
        if self.scale_in_train:
            all_train = [i for sub in train_ranges for i in sub]
            self.scaler.fit(self.dataset.data[all_train])
        else:
            self.scaler.fit(self.dataset.data)

        # build per-series sliding window datasets
        train_ds = []
        val_ds = []
        test_ds = []
        for tr, vr, te in zip(train_ranges, val_ranges, test_ranges):
            sub_train = TimeseriesSubset(self.dataset, tr)
            sub_val = TimeseriesSubset(self.dataset, vr)
            sub_test = TimeseriesSubset(self.dataset, te)
            train_ds.append(
                MultiStepTimeFeatureSet(
                    sub_train,
                    scaler=self.scaler,
                    time_enc=self.time_enc,
                    window=self.window,
                    horizon=self.horizon,
                    steps=self.steps,
                    freq=self.freq,
                    single_variate=self.single_variate,
                    scaler_fit=False,
                )
            )
            val_ds.append(
                MultiStepTimeFeatureSet(
                    sub_val,
                    scaler=self.scaler,
                    time_enc=self.time_enc,
                    window=self.window,
                    horizon=self.horizon,
                    steps=self.steps,
                    freq=self.freq,
                    single_variate=self.single_variate,
                    scaler_fit=False,
                )
            )
            test_ds.append(
                MultiStepTimeFeatureSet(
                    sub_test,
                    scaler=self.scaler,
                    time_enc=self.time_enc,
                    window=self.window,
                    horizon=self.horizon,
                    steps=self.steps,
                    freq=self.freq,
                    single_variate=self.single_variate,
                    scaler_fit=False,
                )
            )

        # concatenate
        self.train_dataset = ConcatDataset(train_ds)
        self.val_dataset = ConcatDataset(val_ds)
        self.test_dataset = ConcatDataset(test_ds)

        # dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_worker,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            drop_last=False,
        )

        self.num_features = self.data.shape[1]
        self.length = self.data.shape[0]

        # Infer frequency from first series
        freq_str = pd.infer_freq(dates_list[0]['date'])
        if freq_str and freq_str[0] in Freq._value2member_map_:
            # Map to enum
            self.freq = Freq(freq_str[0])
        else:
            # Default to days
            self.freq = Freq.days

        return self.data
