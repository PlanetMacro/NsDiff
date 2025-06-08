import os
import torch
import yaml
import numpy as np
import pandas as pd
from types import SimpleNamespace
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.core import TimeSeriesDataset
from torch_timeseries.utils.timefeatures import time_features
import src.layer.mu_backbone as ns_Transformer
import src.layer.g_backbone as G
from src.models.NsDiff import NsDiff
from src.experiments.NsDiff import dict2namespace, p_sample_loop

# Inference class for NsDiff model
default_scaler = "StandardScaler"
class NsDiffInference:
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        save_dir: str,
        windows: int,
        horizon: int,
        pred_len: int,
        seed: int,
        device: str = "cpu",
        scaler_type: str = default_scaler,
        time_enc: int = 3,
        minisample: int = 1,
    ):
        # Initialize parameters
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.save_dir = save_dir
        self.windows = windows
        self.horizon = horizon
        self.pred_len = pred_len
        self.seed = seed
        self.device = torch.device(device)
        self.time_enc = time_enc
        # Load diffusion configuration
        cfg_path = os.path.join("configs", "nsdiff.yml")
        with open(cfg_path, "r") as f:
            cfg = yaml.unsafe_load(f)
        self.diffusion_config = dict2namespace(cfg)
        # Load dataset
        self.dataset: TimeSeriesDataset = parse_type(dataset_type, globals())(
            root=data_path
        )
        # No scaling, use identity transform
        self.scaler = SimpleNamespace(transform=lambda x: x, inverse_transform=lambda x: x)
        # Build models
        label_len = windows // 2
        args = {
            "seq_len": windows,
            "device": self.device,
            "pred_len": pred_len,
            "label_len": label_len,
            "features": "M",
            "beta_start": cfg.get("beta_start"),
            "beta_end": cfg.get("beta_end"),
            "enc_in": self.dataset.num_features,
            "dec_in": self.dataset.num_features,
            "c_out": self.dataset.num_features,
            "d_model": cfg.get("d_model"),
            "n_heads": cfg.get("n_heads"),
            "e_layers": cfg.get("e_layers"),
            "d_layers": cfg.get("d_layers"),
            "d_ff": cfg.get("d_ff"),
            "moving_avg": cfg.get("moving_avg"),
            "timesteps": cfg.get("timesteps"),
            "factor": cfg.get("factor"),
            "distil": cfg.get("distil"),
            "beta_schedule": cfg.get("beta_schedule", "linear"),
            "embed": cfg.get("embed", "timeF"),
            "dropout": cfg.get("dropout"),
            "activation": cfg.get("activation"),
            "output_attention": False,
            "do_predict": True,
            "k_z": cfg.get("k_z"),
            "k_cond": cfg.get("k_cond"),
            "p_hidden_dims": cfg.get("p_hidden_dims"),
            "freq": self.dataset.freq,
            # G and sigma params
            "CART_input_x_embed_dim": cfg.get("CART_input_x_embed_dim"),
            "p_hidden_layers": cfg.get("p_hidden_layers"),
            "d_z": cfg.get("d_z"),
            "diffusion_config_dir": cfg.get("diffusion_config_dir"),
        }
        self.args = SimpleNamespace(**args)
        # Initialize diffusion model (NsDiff) and conditional predictors
        self.model = NsDiff(self.args, self.device).to(self.device)
        self.cond_pred_model = ns_Transformer.Model(self.args).float().to(self.device)
        self.cond_pred_model_g = G.SigmaEstimation(
            windows,
            pred_len,
            self.dataset.num_features,
            cfg.get("CART_input_x_embed_dim"),
            cfg.get("rolling_length", windows // 2),
        ).float().to(self.device)
        # Load pretrained weights
        # Determine runs directory: either save_dir/runs or save_dir/results/runs
        runs_root = os.path.join(save_dir, "runs")
        if not os.path.isdir(runs_root):
            runs_root = os.path.join(save_dir, "results", "runs")
        # Expect weights at: <runs_root>/NsDiff4/{dataset_type}/w{windows}h{horizon}s{pred_len}/{seed}/
        run_dir = os.path.join(
            runs_root,
            "NsDiff4",
            dataset_type,
            f"w{windows}h{horizon}s{pred_len}",
            str(seed),
        )
        self.model.load_state_dict(
            torch.load(os.path.join(run_dir, "model.pth"), map_location=self.device)
        )
        self.cond_pred_model.load_state_dict(
            torch.load(
                os.path.join(run_dir, "cond_pred_model.pth"), map_location=self.device
            )
        )
        self.cond_pred_model_g.load_state_dict(
            torch.load(
                os.path.join(run_dir, "cond_pred_model_g.pth"), map_location=self.device
            )
        )
        self.model.eval()
        self.cond_pred_model.eval()
        self.cond_pred_model_g.eval()
        self.minisample = minisample
        self.label_len = label_len

    def infer(self, window: np.ndarray) -> np.ndarray:
        """
        Perform inference on a raw time-series window.
        Args:
            window: np.ndarray of shape (windows, num_features) or (windows,)
        Returns:
            np.ndarray: predicted values of shape (pred_len, num_features)
        """
        x = np.asarray(window)
        if x.ndim == 1:
            x = x[:, None]
        # Scale input
        scaled_x = self.scaler.transform(x)
        # Time encoding for x
        dates = self.dataset.dates.iloc[-self.windows:].reset_index(drop=True)
        x_mark = time_features(dates, self.time_enc, self.dataset.freq)
        # Time encoding for y (future)
        # Generate future dates
        # Map freq to pandas freq strings
        freq_map = {"s": "S", "t": "T", "h": "H", "d": "D", "m": "M", "y": "A"}
        pd_freq = freq_map.get(self.dataset.freq, self.dataset.freq)
        last_date = dates["date"].iloc[-1]
        future_all = pd.date_range(start=last_date, periods=self.pred_len + 1, freq=pd_freq)
        future_dates = future_all[1:]
        future_df = pd.DataFrame({"date": future_dates})
        y_mark = time_features(future_df, self.time_enc, self.dataset.freq)
        # Prepare tensors
        batch_x = torch.tensor(scaled_x, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch_x_mark = torch.tensor(x_mark, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch_y_mark = torch.tensor(y_mark, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Dummy batch_y
        batch_y = torch.zeros(
            (1, self.pred_len, self.dataset.num_features),
            dtype=torch.float32,
            device=self.device,
        )
        # Run inference
        with torch.no_grad():
            outs = self._process_val_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
        # outs shape: (1, pred_len, num_features, n_z_samples)
        # Return mean across samples
        return outs.mean(dim=-1).squeeze(0).cpu().numpy()

    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # Based on NsDiff._process_val_batch logic
        b = batch_x.shape[0]
        minisample = self.minisample
        # concat past and future marks for decoder input
        batch_y_mark_input = torch.cat(
            [batch_x_mark[:, -self.label_len :, :], batch_y_mark], dim=1
        )
        # Decoder input: zeros for future steps
        dec_pred = torch.zeros(
            [b, self.pred_len, self.dataset.num_features], device=self.device
        )
        dec_label = batch_x[:, -self.label_len :, :]
        dec_inp = torch.cat([dec_label, dec_pred], dim=1)
        # Conditional predictions
        y0_hat, _ = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark_input
        )
        gx = self.cond_pred_model_g(batch_x)
        # Random timesteps
        n = b
        t = torch.randint(
            low=0, high=self.model.num_timesteps, size=(n // 2 + 1,), device=self.device
        )
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]
        # Sampling
        preds = []
        for _ in range(self.diffusion_config.testing.n_z_samples // minisample):
            # tile inputs
            rep = minisample
            y0 = y0_hat.repeat(rep, 1, 1, 1).transpose(0, 1).flatten(0, 1)
            yT = y0
            x_t = batch_x.repeat(rep, 1, 1, 1).transpose(0, 1).flatten(0, 1)
            xm = batch_x_mark.repeat(rep, 1, 1, 1).transpose(0, 1).flatten(0, 1)
            gx_t = gx.repeat(rep, 1, 1, 1).transpose(0, 1).flatten(0, 1)
            # draw samples
            gen_box = []
            for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                    seq = p_sample_loop(
                        self.model,
                        x_t,
                        xm,
                        y0,
                        gx_t,
                        yT,
                        self.model.num_timesteps,
                        self.model.alphas,
                        self.model.one_minus_alphas_bar_sqrt,
                        self.model.alphas_cumprod,
                        self.model.alphas_cumprod_sum,
                        self.model.alphas_cumprod_prev,
                        self.model.alphas_cumprod_sum_prev,
                        self.model.betas_tilde,
                        self.model.betas_bar,
                        self.model.betas_tilde_m_1,
                        self.model.betas_bar_m_1,
                    )
                gen = seq[self.model.num_timesteps].reshape(
                    b,
                    minisample,
                    self.pred_len,
                    self.dataset.num_features,
                )
                gen_box.append(gen.detach().cpu())
            out = torch.cat(gen_box, dim=1)  # B, n_z_samples, pred_len, N? adjust
            preds.append(out)
        preds = torch.cat(preds, dim=1)  # B, total_samples, pred_len, N
        # reorder: (B, pred_len, N, samples)
        return preds.permute(0, 2, 3, 1)  # shape B, O, N, n_z_samples
