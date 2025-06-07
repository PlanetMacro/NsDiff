import os
import pandas as pd
import numpy as np
from torch_timeseries.core import TimeSeriesDataset, Freq


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
