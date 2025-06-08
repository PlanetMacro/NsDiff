#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch_timeseries.utils.timefeatures import time_features

# Ensure project src is in PYTHONPATH
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.NsDiffInference import NsDiffInference

def parse_args():
    parser = argparse.ArgumentParser(description="Forecast time series with NsDiffInference")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root path of the dataset (folder containing dataset files)")
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="Dataset type (e.g., ETTH1, ETTH2, Custom)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Root path where model runs are saved")
    parser.add_argument("--windows", type=int, required=True,
                        help="Length of input window")
    parser.add_argument("--horizon", type=int, required=True,
                        help="Horizon (used for naming/loading) but not affecting inference length)")
    parser.add_argument("--pred_len", type=int, required=True,
                        help="Number of steps to predict")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to choose start index")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference (cpu or cuda)")
    parser.add_argument("--scaler_type", type=str, default="StandardScaler",
                        help="Scaler type to normalize data (e.g., StandardScaler)")
    parser.add_argument("--time_enc", type=int, default=3,
                        help="Time encoding dimension")
    parser.add_argument("--minisample", type=int, default=1,
                        help="Minisample factor for diffusion inference")
    parser.add_argument("--output", type=str, default="forecast.png",
                        help="Output image file path")
    return parser.parse_args()


def main():
    args = parse_args()
    # Initialize inference object
    inferer = NsDiffInference(
        data_path=args.data_path,
        dataset_type=args.dataset_type,
        save_dir=args.save_dir,
        windows=args.windows,
        horizon=args.horizon,
        pred_len=args.pred_len,
        seed=args.seed,
        device=args.device,
        scaler_type=args.scaler_type,
        time_enc=args.time_enc,
        minisample=args.minisample
    )
    # Load full dataset as pandas.DataFrame
    data = inferer.dataset.data
    # Choose random start index for window
    np.random.seed(args.seed)
    total_len = len(data)
    max_start = total_len - args.windows - args.pred_len
    if max_start < 0:
        raise ValueError(f"Not enough data: total={total_len}, windows={args.windows}, pred_len={args.pred_len}")
    start = np.random.randint(0, max_start + 1)
    # Extract history and true future
    history_df = data.iloc[start : start + args.windows]
    future_df = data.iloc[start + args.windows : start + args.windows + args.pred_len]
    history = history_df.values
    true_future = future_df.values
    # Prepare batch for inference
    x = history
    if x.ndim == 1:
        x = x[:, None]
    scaled_x = inferer.scaler.transform(x)
    # Time features for history
    hist_dates = inferer.dataset.dates.iloc[start : start + args.windows].reset_index(drop=True)
    x_mark = time_features(hist_dates, args.time_enc, inferer.dataset.freq)
    # Time features for future
    last_date = hist_dates["date"].iloc[-1]
    freq_map = {"s": "S", "t": "T", "h": "H", "d": "D", "m": "M", "y": "A"}
    pd_freq = freq_map.get(inferer.dataset.freq, inferer.dataset.freq)
    future_dates = pd.date_range(start=last_date, periods=args.pred_len + 1, freq=pd_freq)[1:]
    future_dates_df = pd.DataFrame({"date": future_dates})
    y_mark = time_features(future_dates_df, args.time_enc, inferer.dataset.freq)
    # Tensorize inputs
    batch_x = torch.tensor(scaled_x, dtype=torch.float32).unsqueeze(0).to(inferer.device)
    batch_x_mark = torch.tensor(x_mark, dtype=torch.float32).unsqueeze(0).to(inferer.device)
    batch_y_mark = torch.tensor(y_mark, dtype=torch.float32).unsqueeze(0).to(inferer.device)
    batch_y = torch.zeros((1, args.pred_len, inferer.dataset.num_features),
                          dtype=torch.float32).to(inferer.device)
    # Run inference to get sample distribution
    with torch.no_grad():
        preds = inferer._process_val_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
    # preds shape: (1, pred_len, num_features, n_samples)
    preds = preds.squeeze(0).cpu().numpy()  # shape: pred_len, num_features, n_samples
    # Plot results
    n_features = preds.shape[1]
    n_samples = preds.shape[2]
    fig, axes = plt.subplots(n_features, 1, figsize=(8, 4 * n_features))
    if n_features == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        # History
        ax.plot(range(args.windows), history[:, i], label="History", color="black")
        # True future
        ax.plot(range(args.windows, args.windows + args.pred_len),
                true_future[:, i], label="True Future", color="blue", linewidth=2)
        # Predicted futures
        for s in range(n_samples):
            ax.plot(range(args.windows, args.windows + args.pred_len),
                    preds[:, i, s], color="red", alpha=0.1)
        ax.set_title(f"Feature {i}")
        ax.legend()
    plt.tight_layout()
    # Save image
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output)
    print(f"Forecast plot saved to {args.output}")

if __name__ == "__main__":
    main()
