from __future__ import annotations
from dataclasses import dataclass
from tqdm import tqdm
import torch
from torch import nn
import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf

from src.experiments.prob_forecast import ProbForecastExp
from src.datasets import Custom 

from torch_timeseries.utils.parse_type import parse_type
from src.experiments.NsDiffInference import NsDiffInference


# ============================================================================
# 1.  MOCK building blocks you will later replace
# ----------------------------------------------------------------------------
class MOCK_NET(nn.Module):
    """A *very* small network that only serves as a placeholder.

    Architecture: (Last-time-step) ŌåÆ Linear ŌåÆ ReLU ŌåÆ Linear.

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
    is *not* forecasting (e.g. `ImputationExp`, `AnomalyExp`, ŌĆ”).
    """
    use_gpu: bool = True  # default: try GPU if available


    def __post_init__(self):
        # Ensure backward compat again
        if not hasattr(np, "Inf"):
            np.Inf = np.inf
        # Select device
        if self.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"


    # --- identification strings shown in logs / wandb ----------------------
    model_type: str = "MOCK_NET"         # appears in filenames & wandb runs
    loss_func_type: str = "mock"          # just a placeholder for readability
    # Extra parameters required for NsDiffInference
    horizon: int = 24
    seed: int = 42
    minisample: int = 1
    freq: str = "d"


    # ---------------------------------------------------------------------
    # 2.1  Loss ŌĆō overwrite `_init_loss_func`
    # ---------------------------------------------------------------------

    def _init_loss_func(self) -> None:  # noqa: D401
        """Replace parent loss initialisation by our `MOCK_LOSS`."""
        self.loss_func = MOCK_LOSS()

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

    def _init_model(self) -> None:
        """Create the model *after* `self.dataset` has been prepared.

        Access to `self.dataset.num_features` allows the network to know how
        many variables it has to predict.  All other hyper-parameters come
        from the dataclass (`hidden`, etc.).
        """

        input_dim = self.dataset.num_features
        self.model = MOCK_NET(input_dim, self.hidden).to(self.device)

        # Instantiate pre-trained NsDiffInference (required)
        self.pathSpaceSampler = NsDiffInference(
            dataset_type=self.dataset_type,
            windows=self.windows,
            horizon=self.horizon,
            pred_len=self.pred_len,
            num_features=input_dim,
            seed=self.seed,
            device=self.device,
            minisample=self.minisample,
            freq=self.freq,
        )

        # Use parent class helper to create optimizer
        super()._init_optimizer()


    # ---------------------------------------------------------------------
    # Training- & validation-batch helpers required by ProbForecastExp
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


    # ---------------------------------------------------------------------
    # 2.3c  Evaluation ŌĆō keep full prediction shape for probabilistic metrics
    # ---------------------------------------------------------------------
    def _evaluate(self, dataloader):  # noqa: C901
        """Copy of ``ProbForecastExp._evaluate`` without the shape squeeze.

        The parent class flattens predictions when ``pred_len==1`` which breaks
        CRPS/QICE/etc.  Here we always feed the *full* tensor to the metrics ŌĆō
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
    # 2.4  Core logic ŌĆō overwrite `_process_one_batch`
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
            batch_y  : (B, steps, N)   ŌĆō not used here, but provided
            origin_y : (B, T, N)       ŌĆō *unscaled* ground truth (if needed)

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
        # and one sample (S=1) ŌåÆ add singleton dims.
        pred = pred_deterministic.unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)

        # Target ŌåÆ last time step of the (already *unscaled*) ground truth.
        true = origin_y[:, -1, :].unsqueeze(1)   # (B, 1, N)

        return pred, true

    def _train(self):
        self.model.train()

        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device).float()
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                # generate path space samples
                with torch.no_grad():
                    paths = self.pathSpaceSampler.infer(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc)

                loss = self._process_train_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )                
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
                # loss.backward()

                progress_bar.update(batch_x.size(0))
                
                #train_loss.append(loss.item())
                #progress_bar.set_postfix(
                #    loss=loss.item(),
                #    lr=self.model_optim.param_groups[0]["lr"],
                #    epoch=self.current_epoch,
                #    refresh=True,
                #)
                #self.model_optim.step()
                #self.model_optim.zero_grad()
                

        self.model.eval()
        return train_loss


# ============================================================================
# 3.  Convenience factory ŌĆō *optional*
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
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minisample", type=int, default=1)
    parser.add_argument("--freq", type=str, default="d")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    args, _ = parser.parse_known_args()

    # Build the experiment from CLI arguments.  All remaining defaults come
    # from `ForecastExp` & our dataclasses.
    exp = NsTradeExp(
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        windows=args.windows,
        pred_len=args.pred_len,
        horizon=args.horizon,
        seed=args.seed,
        minisample=args.minisample,
        freq=args.freq,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_gpu=args.gpu,
    )

    # Run one seed only for brevity.  If you need multiple seeds simply loop
    # over them and call `exp.run(seed)`.
    exp.run(seed=42)
