"""
NsTrade – A minimal example showing how to write a fully-featured training
loop with `torch_timeseries` while **keeping the code base of this repo
unchanged**.

The goal is **pedagogical**: it demonstrates how you can
  • subclass an existing experiment class (`ForecastExp`),
  • plug-in your own model,
  • plug-in your own loss function,
  • add extra hyper-parameters via a `@dataclass`,
all without touching anything inside the `torch_timeseries` package.

Everything marked with the word "MOCK" is *dummy* code that you will replace
with your real model / loss later on – explanatory comments tell you *where*
and *how* to do that.
"""
from __future__ import annotations

# --------------- standard library ---------------
from dataclasses import dataclass

# --------------- third-party ---------------
import torch
from torch import nn
from torch.optim import Adam

# --------------- torch_timeseries ---------------
# `ForecastExp` already implements: data loading, metric tracking, early
# stopping, logging, etc.  We only need to supply a model + loss + the core
# forward method.
import numpy as np
# Ensure backward‐compat constant for older utils expecting `np.Inf`
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from src.experiments.prob_forecast import ProbForecastExp  # use the more feature-rich experiment
from src.datasets import Custom  # ensure 'Custom' dataset is in globals for parse_type

from torch_timeseries.utils.parse_type import parse_type  # reuse utility

# ============================================================================
# 1.  MOCK building blocks you will later replace
# ----------------------------------------------------------------------------
class MOCK_NET(nn.Module):
    """A *very* small network that only serves as a placeholder.

    Architecture: (Last-time-step) → Linear → ReLU → Linear.

    Replace this class with **your real forecasting model**.  All you have to
    keep is the typical PyTorch `forward()` signature.
    """

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, N)
        # Here we take only the *last* time step.  Real models will of course
        # use the whole sequence.
        return self.net(x[:, -1, :])  # (B, N)


class MOCK_LOSS(nn.Module):
    """Simple MSE written explicitly so that the example shows *custom* code."""

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        return torch.mean((pred - true) ** 2)


# Hyper-parameters that are specific to our mock model.
# Add anything you need for the real network.
@dataclass
class MOCK_Parameters:
    hidden: int = 64  # hidden size of `MOCK_NET` (change or extend freely)


# ============================================================================
# 2.  The actual experiment class
# ----------------------------------------------------------------------------
@dataclass
class NsTradeExp(ProbForecastExp, MOCK_Parameters):  # type: ignore[misc]
    """Custom experiment that plugs `MOCK_NET` + `MOCK_LOSS` into the generic
    training/validation/testing loop provided by `ProbForecastExp`.  This gives you all probability-forecasting metrics and appropriate dataloaders out of the box.

    Inherit from another `torch_timeseries.experiments.*` class if your task
    is *not* forecasting (e.g. `ImputationExp`, `AnomalyExp`, …).
    """
    use_gpu: bool = False  # ensure every run stays on CPU


    def __post_init__(self):
        # Ensure backward compat again
        if not hasattr(np, "Inf"):
            np.Inf = np.inf
        # After dataclass init, lock device to CPU
        self.device = "cpu"


    # `ForecastExp` already defines plenty of attributes (batch_size, lr, …)
    # Below we only set the ones we *want* to overwrite / add.

    # --- identification strings shown in logs / wandb ----------------------
    model_type: str = "MOCK_NET"         # appears in filenames & wandb runs
    loss_func_type: str = "mock"          # just a placeholder for readability

    # ---------------------------------------------------------------------
    # 2.1  Loss – overwrite `_init_loss_func`
    # ---------------------------------------------------------------------

    def _init_loss_func(self) -> None:  # noqa: D401
        """Replace parent loss initialisation by our `MOCK_LOSS`."""
        self.loss_func = MOCK_LOSS()

    # ---------------------------------------------------------------------
    # 2.1b  Metrics – keep it simple for this CPU demo
    # ---------------------------------------------------------------------


    def _init_metrics(self):  # noqa: D401
        """Reuse full probabilistic metrics from parent but downsize pool."""
        super()._init_metrics()
        import torch.multiprocessing as mp
        if hasattr(self, "task_pool"):
            try:
                self.task_pool.terminate()
                self.task_pool.join()
            except Exception:
                pass
        ctx = mp.get_context("spawn")
        self.task_pool = ctx.Pool(processes=1)

    # ---------------------------------------------------------------------
    # 2.2  Model – overwrite `_init_model`
    # ---------------------------------------------------------------------

    def _init_model(self) -> None:
        """Create the model *after* `self.dataset` has been prepared.

        Access to `self.dataset.num_features` allows the network to know how
        many variables it has to predict.  All other hyper-parameters come
        from the dataclass (`hidden`, etc.).
        """

        input_dim = self.dataset.num_features

        # Replace by your REAL model, e.g.
        #   self.model = Informer(
        #       enc_in=input_dim, dec_in=input_dim, c_out=input_dim, …
        #   ).to(self.device)
        self.model = MOCK_NET(input_dim, self.hidden).to(self.device)

        # You can of course reuse `_init_optimizer` from the parent class, but
        # here we show how to plug your own.  Replace `Adam` by whatever you
        # need.
        self.model_optim = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )

    # ---------------------------------------------------------------------
    # 2.3  Optional – skip `_init_optimizer` because we initialise it above
    # ---------------------------------------------------------------------

    def _init_optimizer(self) -> None:  # noqa: D401
        """Optimizer already defined in `_init_model`."""
        pass
    # ---------------------------------------------------------------------
    # 2.3  Training- & validation-batch helpers required by ProbForecastExp
    # ---------------------------------------------------------------------

    def _process_train_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_date_enc: torch.Tensor,
        batch_y_date_enc: torch.Tensor,
    ):
        """Required by `ProbForecastExp._train`.

        We make a *single-step* prediction using the last time step of
        `batch_y` as target.
        """
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()

        pred = self.model(batch_x)               # (B, N)
        true = batch_y[:, -1, :].to(self.device)  # last step
        return pred, true

    def _process_val_batch(
        self,
        batch_x: torch.Tensor,
        batch_origin_x: torch.Tensor,
        batch_x_date_enc: torch.Tensor,
        batch_y_date_enc: torch.Tensor,
    ):
        """Used during validation / testing phases by `ProbForecastExp`."""

        batch_x = batch_x.to(self.device).float()
        batch_origin_x = batch_origin_x.to(self.device).float()

        pred = self.model(batch_x)
        # For validation we again compare to the last step of the *unscaled*
        # original sequence.
        true = batch_origin_x[:, -1, :]
        return pred, true


        # already done in `_init_model`, so we *pass* here.
    # ---------------------------------------------------------------------
    # 2.3c  Evaluation – keep full prediction shape for probabilistic metrics
    # ---------------------------------------------------------------------
    def _evaluate(self, dataloader):  # noqa: C901
        """Copy of ``ProbForecastExp._evaluate`` without the shape squeeze.

        The parent class flattens predictions when ``pred_len==1`` which breaks
        CRPS/QICE/etc.  Here we always feed the *full* tensor to the metrics –
        our ``_process_one_batch`` already returns the expected 4-D/3-D shapes.
        """
        from tqdm import tqdm
        import torch
        from src.experiments.prob_forecast import update_metrics  # same helper

        self.model.eval()
        self.metrics.reset()
        results = []
        with tqdm(total=len(dataloader.dataset)) as bar:
            for (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in dataloader:
                # send to device
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                origin_x = origin_x.to(self.device)
                origin_y = origin_y.to(self.device)
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                preds, truths = self._process_one_batch(
                    batch_x,
                    batch_y,
                    origin_x,
                    origin_y,
                    batch_x_date_enc,
                    batch_y_date_enc,
                )

                if self.invtrans_loss:
                    preds = self.scaler.inverse_transform(preds)
                    truths = origin_y

                results.append(
                    self.task_pool.apply_async(
                        update_metrics,
                        (
                            preds.contiguous().cpu().detach(),
                            truths.contiguous().cpu().detach(),
                            self.metrics,
                        ),
                    )
                )
                bar.update(batch_x.size(0))

        for r in results:
            r.get()
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

        pass

    # ---------------------------------------------------------------------
    # 2.4  Core logic – overwrite `_process_one_batch`
    # ---------------------------------------------------------------------

    def _process_one_batch(  # type: ignore[override]
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_origin_x: torch.Tensor,
        batch_origin_y: torch.Tensor,
        batch_x_date_enc: torch.Tensor,
        batch_y_date_enc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """How to turn a dataloader batch into *predictions* & *targets*.

        The default shape conventions in `ForecastExp` are:
            batch_x  : (B, T, N)
            batch_y  : (B, steps, N)   – not used here, but provided
            origin_y : (B, T, N)       – *unscaled* ground truth (if needed)

        A *real* sequence model would usually take *both* `batch_x` and
        `batch_y` (teacher-forcing).  For the sake of clarity we demonstrate a
        *single-step* prediction based on the past `batch_x` only.
        """

        # Move everything we actually use onto the right device
        batch_x = batch_x.to(self.device).float()      # input sequence
        origin_y = batch_origin_y.to(self.device).float()

        # -----------------------------------------------------------------
        # Forward pass through our model
        # -----------------------------------------------------------------
        pred_deterministic = self.model(batch_x)  # (B, N)
        # For probabilistic metrics expect (B, O, N, S).  We have only one step (O=1)
        # and one sample (S=1) → add singleton dims.
        pred = pred_deterministic.unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)

        # Target → last time step of the (already *unscaled*) ground truth.
        true = origin_y[:, -1, :].unsqueeze(1)   # (B, 1, N)

        return pred, true


# ============================================================================
# 3.  Convenience factory – *optional*
# ----------------------------------------------------------------------------
# With the following helper you can launch training from the command line:
#
#   python -m src.experiments.NsTrade \
#       --dataset_type ETTh1 \
#       --windows 336          \
#       --pred_len 96          \
#       --epochs 10
#
# Feel free to delete this section if you already have your own runner
# infrastructure (e.g. Hydra, Lightning CLI, or shell scripts).
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the NsTrade experiment")
    parser.add_argument("--dataset_type", required=True)
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--windows", type=int, default=336)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args, _ = parser.parse_known_args()

    # Build the experiment from CLI arguments.  All remaining defaults come
    # from `ForecastExp` & our dataclasses.
    exp = NsTradeExp(
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        windows=args.windows,
        pred_len=args.pred_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Run one seed only for brevity.  If you need multiple seeds simply loop
    # over them and call `exp.run(seed)`.
    exp.run(seed=42)
