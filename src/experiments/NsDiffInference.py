"""
NsDiffInference
~~~~~~~~~~~~~~~
Run inference with a trained NsDiff diffusion model (plus its conditional
predictors F and G) for any dataset / seed.

Directory layout expected
-------------------------
results/runs/
│
├─ NsDiff4/<dataset>/w{w}h{h}s{s}/{seed}/model.pth
├─ F       /<dataset>/w{w}h{h}s{s}/{seed}/best_model.pth
│          └─ args.json            <-- hyper-params for F
└─ G       /<dataset>/w{w}h{h}s{s}/{seed}/best_model.pth
           └─ args.json            <-- hyper-params for G
"""

import json
import os
from types import SimpleNamespace

import yaml
import torch
import pandas as pd
import numpy as np

from src.models.NsDiff import NsDiff
import src.layer.mu_backbone as F                        # predictor
import src.layer.g_backbone as G                         # sigma estimator
from src.experiments.NsDiff import dict2namespace
from src.layer.nsdiff_utils import p_sample_loop


class NsDiffInference:
    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        dataset_type: str,
        windows: int,
        horizon: int,
        pred_len: int,
        num_features: int,
        seed: int,
        device: str = "cpu",
        minisample: int = 1,
        freq: str = "d",
    ):
        # ---------- base attributes ----------
        self.windows = windows
        self.horizon = horizon
        self.pred_len = pred_len
        self.num_features = num_features
        self.seed = seed
        self.device = torch.device(device)
        self.minisample = minisample
        self.label_len = windows // 2

        # ---------- load global diffusion config ----------
        with open(os.path.join("configs", "nsdiff.yml"), "r") as f:
            cfg = yaml.safe_load(f)
        self.diffusion_config = dict2namespace(cfg)

        # ---------- assemble arg namespace ----------
        args = SimpleNamespace(
            seq_len=windows,
            device=self.device,
            pred_len=pred_len,
            label_len=self.label_len,
            features="M",
            beta_start=cfg["beta_start"],
            beta_end=cfg["beta_end"],
            enc_in=num_features,
            dec_in=num_features,
            c_out=num_features,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            e_layers=cfg["e_layers"],
            d_layers=cfg["d_layers"],
            d_ff=cfg["d_ff"],
            moving_avg=cfg["moving_avg"],
            timesteps=cfg["timesteps"],
            factor=cfg["factor"],
            distil=cfg["distil"],
            beta_schedule=cfg.get("beta_schedule", "linear"),
            embed=cfg.get("embed", "timeF"),
            dropout=cfg["dropout"],
            activation=cfg["activation"],
            output_attention=False,
            do_predict=True,
            k_z=cfg["k_z"],
            k_cond=cfg["k_cond"],
            p_hidden_dims=cfg["p_hidden_dims"],
            freq=freq,
            CART_input_x_embed_dim=cfg["CART_input_x_embed_dim"],
            p_hidden_layers=cfg["p_hidden_layers"],
            d_z=cfg["d_z"],
            diffusion_config_dir=cfg["diffusion_config_dir"],
        )
        self.args = args                         # keep for callers

        # ---------- instantiate NsDiff + F ----------
        self.NsDiff = NsDiff(args, self.device).to(self.device)
        self.F = F.Model(args).float().to(self.device)

        # ---------- locate run directories ----------
        runs_root = os.path.join(os.getcwd(), "results", "runs")
        ns_dir = os.path.join(
            runs_root,
            "NsDiff4",
            dataset_type,
            f"w{windows}h{horizon}s{pred_len}",
            str(seed),
        )
        f_dir = os.path.join(
            runs_root,
            "F",
            dataset_type,
            f"w{windows}h{horizon}s{pred_len}",
            str(seed),
        )
        g_dir = os.path.join(
            runs_root,
            "G",
            dataset_type,
            f"w{windows}h{horizon}s{pred_len}",
            str(seed),
        )

        # ---------- load NsDiff + F checkpoints ----------
        self.NsDiff.load_state_dict(
            torch.load(os.path.join(ns_dir, "model.pth"), map_location=self.device)
        )
        self.F.load_state_dict(
            torch.load(os.path.join(f_dir, "best_model.pth"), map_location=self.device)
        )

        # ---------- recreate & load G exactly as trained ----------
        with open(os.path.join(g_dir, "args.json"), "r") as f:
            g_args = json.load(f)

        hidden_dim  = g_args.get("hidden_size", 512)
        kernel_size = (
            g_args.get("kernel_size") or           # some training scripts use this key
            g_args.get("rolling_length")           # others use rolling_length
        )
        if kernel_size is None:
            raise ValueError(
                "Could not infer kernel_size from G args.json; "
                "expected keys 'kernel_size' or 'rolling_length'."
            )

        self.G = G.SigmaEstimation(
            windows, pred_len, num_features, hidden_dim, kernel_size
        ).float().to(self.device)

        self.G.load_state_dict(
            torch.load(os.path.join(g_dir, "best_model.pth"), map_location=self.device)
        )

        # ---------- eval mode ----------
        self.NsDiff.eval()
        self.F.eval()
        self.G.eval()

        print("NsDiffInference ready. Args snapshot:")
        print(self.args)

    # ------------------------------------------------------------------ #
    # add your own sampling logic here                                    #
    # ------------------------------------------------------------------ #
    def infer(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        
        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)        
        
        # f(X)
        y_0_hat_batch, _ = self.F(batch_x, batch_x_mark, dec_inp,batch_y_mark_input)
        # g(X)
        gx = self.G(batch_x)
        
        repeat_n = self.minisample
        y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
        y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
        y_T_mean_tile = y_0_hat_tile
        x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
        x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

        x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
        x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

        gx_tile = gx.repeat(repeat_n, 1, 1, 1)
        gx_tile = gx_tile.transpose(0, 1).flatten(0, 1).to(self.device)        

        y_tile_seq = p_sample_loop(self.NsDiff, x_tile, x_mark_tile, y_0_hat_tile, gx_tile, y_T_mean_tile,
                            self.NsDiff.num_timesteps,
                            self.NsDiff.alphas, 
                            self.NsDiff.one_minus_alphas_bar_sqrt,
                            self.NsDiff.alphas_cumprod, 
                            self.NsDiff.alphas_cumprod_sum,
                            self.NsDiff.alphas_cumprod_prev, 
                            self.NsDiff.alphas_cumprod_sum_prev,
                            self.NsDiff.betas_tilde, 
                            self.NsDiff.betas_bar,
                            self.NsDiff.betas_tilde_m_1, 
                            self.NsDiff.betas_bar_m_1,
                            )
        return y_tile_seq[-1]
        #print(f"Num_timesteps: {self.NsDiff.num_timesteps}")
        #print(f"shape x_tile : {x_tile.shape}")                    
        #print(f"shape y_tile: {y_tile.shape}")
