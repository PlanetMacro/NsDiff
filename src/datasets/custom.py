import os
import pandas as pd
import numpy as np
from torch_timeseries.core import TimeSeriesDataset, Freq

class CustomCsv(TimeSeriesDataset):
    """
    Custom CSV dataset loader. Expects a CSV file in the data/custom directory.
    The first column is parsed as datetime, and the remaining columns as features.
    Splits and batching are handled by SlidingWindowTS.
    """
    name: str = 'custom'

    def download(self):
        # No download needed; data should exist in data/custom
        pass

    def _process(self):
        # No pre-processing step
        pass

    def _load(self) -> np.ndarray:
        # Locate CSV file in data/custom
        custom_dir = os.path.join(self.root, 'custom')
        if not os.path.isdir(custom_dir):
            raise FileNotFoundError(f"Custom CSV directory not found: {custom_dir}")
        csv_files = [f for f in os.listdir(custom_dir) if f.lower().endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {custom_dir}")
        # Use first CSV file found
        file_path = os.path.join(custom_dir, csv_files[0])
        # Read CSV, parse first column as datetime
        df = pd.read_csv(file_path, parse_dates=[0])
        if df.shape[1] < 2:
            raise ValueError(f"Custom CSV must contain at least two columns (time + features), got {df.shape}")
        # Assign DataFrame and data arrays
        self.df = df
        # Dates: DataFrame with single column 'date'
        times = df.iloc[:, 0]
        self.dates = pd.DataFrame({'date': times})
        # Data values: all columns except first
        self.data = df.iloc[:, 1:].to_numpy()
        # Set attributes
        self.num_features = self.data.shape[1]
        self.length = self.data.shape[0]
        # Infer frequency from datetime index

        freq_str = pd.infer_freq(times)
        mapping = {'S': 's', 'T': 't', 'H': 'h', 'D': 'd', 'M': 'm', 'A': 'y'}
        if freq_str and freq_str[0] in mapping:
            self.freq = mapping[freq_str[0]]
        else:
            # Default to daily if unable to infer
            self.freq = 'd'
        return self.data
