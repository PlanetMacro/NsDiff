#!/usr/bin/env python3
"""
Main script for running inference with the NsDiffInference class.
"""
import argparse
import numpy as np
import torch 
from src.experiments.NsDiffInference import NsDiffInference
from torch_timeseries.dataloader import SlidingWindowTS
from torch_timeseries.utils.parse_type import parse_type


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize NsDiffInference and run a dummy inference pass."
    )
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="Name of the dataset type used at training")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--horizon", type=int, required=True,
                        help="Forecast horizon used during training")
    parser.add_argument("--pred_len", type=int, required=True,
                        help="Number of time steps to predict")
    parser.add_argument("--windows", type=int, required=True,
                        help="Length of the input window for inference")
    parser.add_argument("--num_features", type=int, required=True,
                        help="Number of features in the input time series")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed used during training")
    parser.add_argument("--minisample", type=int, required=True,
                        help="Mini-sample factor for diffusion sampling")
    parser.add_argument("--freq", type=str, required=True,
                        help="Frequency string (e.g. 'd' for daily)")
    parser.add_argument("--batch_size", type=int, default=1,     # OPTIONAL
                        help="Batch size for the dummy tensors")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Instantiate the inference object                                 #
    # ------------------------------------------------------------------ #
    inferencer = NsDiffInference(
        dataset_type=args.dataset_type,
        windows=args.windows,
        horizon=args.horizon,
        pred_len=args.pred_len,
        num_features=args.num_features,
        seed=args.seed,
        device=args.device,
        minisample=args.minisample,
        freq=args.freq,
    )

    dataloader = SlidingWindowTS(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=True,
                shuffle_train=shuffle,
                freq=self.dataset.freq,
                batch_size=self.batch_size,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                num_worker=self.num_worker,
                fast_test=fast_test,
                fast_val=fast_val,
            )

    print("\nNsDiffInference initialized…")

    # ------------------------------------------------------------------ #
    # 2. Generate random dummy tensors and run one forward-pass inference #
    # ------------------------------------------------------------------ #
    B = args.batch_size               # batch size
    T = args.windows                  # input sequence length
    O = args.pred_len                 # output sequence length
    N = args.num_features             # number of (multivariate) features

    # Set RNG for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # (B, T, N) – historical input
    batch_x = torch.randn(B, T, N, device=device)

    # (B, O, N) – ground-truth future window (not used, but keeps signature)
    batch_y = torch.randn(B, O, N, device=device)

    # Time-marker placeholders – here four generic markers (e.g., [month, day, hour, minute])
    num_marks = 3
    batch_x_mark = torch.zeros(B, T, num_marks, device=device)
    batch_y_mark = torch.zeros(B, O, num_marks, device=device)

    # Run inference without tracking gradients
    with torch.no_grad():
        prediction = inferencer.infer(batch_x, batch_y, batch_x_mark, batch_y_mark)

    print("Dummy inference pass complete")


if __name__ == "__main__":
    main()

