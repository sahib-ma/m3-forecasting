from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


def _ensure_numpy(x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _apply_paper_style(figsize=(6.5, 4), dpi: int = 300) -> None:
    """
    Style taken from data science course paper plotting guidelines.
    """
    plt.rcParams.update(
        {
            "figure.figsize": figsize,
            "figure.dpi": dpi,
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
        }
    )

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    )


def _format_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(which="major", linewidth=0.6, alpha=0.12)

    for loc in ("top", "right"):
        ax.spines[loc].set_visible(False)


def plot_loss_curve(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Axes:
    tr = _ensure_numpy(train_loss)
    vl = _ensure_numpy(val_loss)
    if tr.shape != vl.shape:
        raise ValueError("train_loss and val_loss must have the same shape")

    n = tr.shape[0]
    epochs = np.arange(1, n + 1)

    _apply_paper_style()

    fig, ax = plt.subplots()

    ax.plot(epochs, tr, label="Train", linestyle="-", marker=None)
    ax.plot(epochs, vl, label="Validation", linestyle="--", marker=None)

    mark_every = max(1, n // 10)
    ax.plot(epochs[::mark_every], tr[::mark_every], marker="o", linestyle="", alpha=0.9)
    ax.plot(epochs[::mark_every], vl[::mark_every], marker="s", linestyle="", alpha=0.9)

    ax.set_title("Training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    _format_axes(ax)

    idx = int(np.argmin(vl))
    min_epoch = epochs[idx]
    min_val = vl[idx]
    ax.axvline(min_epoch, color="0.8", linewidth=0.8, linestyle=":")
    ax.scatter([min_epoch], [min_val], color="#d95f02", zorder=4)
    ax.annotate(
        f"min val\n{min_val:.4f}\n(ep {int(min_epoch)})",
        xy=(min_epoch, min_val),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", lw=0.6),
    )

    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        fmt = p.suffix.replace(".", "") or "pdf"
        if fmt == "":
            fmt = "pdf"
            p = p.with_suffix(".pdf")
        plt.savefig(p, format=fmt, bbox_inches="tight", dpi=300)

    return ax


def plot_metric_curve(
    train_metric: Sequence[float],
    val_metric: Sequence[float],
    metric_name: str = "Metric",
    **plot_kwargs,
) -> plt.Axes:
    return plot_loss_curve(
        train_metric,
        val_metric,
        ylabel=metric_name,
        title=f"Train / Validation {metric_name}",
        **plot_kwargs,
    )



###############################################################################################
#STL decomposition
from preprocess import decompose, project_components
import pandas as pd

def plot_stl_decomposition_example(
    train: np.ndarray,
    h: int = 18,
    season: int = 12,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot STL decomposition (trend, seasonal, residual) fitted on training data only,
    and extend trend/seasonality into the forecast horizon.

    """
    _apply_paper_style()


    trend, seasonal, resid = decompose(train, season=season)

    trend_f, seas_f = project_components(trend, seasonal, h=h, season=season)

   
    t_train = np.arange(len(train))
    t_future = np.arange(len(train), len(train) + h)


    fig, axes = plt.subplots(4, 1, figsize=(6.5, 6), sharex=True)
    fig.suptitle("STL Decomposition (Train-only fit)")

    # 1. Original series
    axes[0].plot(t_train, train, label="Original (Train)")
    axes[0].axvspan(len(train) - h, len(train), color="0.9", alpha=0.3, label="Forecast horizon")
    axes[0].set_ylabel("Original")
    axes[0].legend(loc="upper left")
    _format_axes(axes[0])

    # 2. Trend
    axes[1].plot(t_train, trend, label="Trend (train fit)")
    axes[1].plot(t_future, trend_f, "--", label="Trend (extrapolated)")
    axes[1].set_ylabel("Trend")
    axes[1].legend(loc="upper left")
    _format_axes(axes[1])

    # 3. Seasonal
    axes[2].plot(t_train, seasonal, label="Seasonal (train fit)")
    axes[2].plot(t_future, seas_f, "--", label="Seasonal (repeated cycle)")
    axes[2].set_ylabel("Seasonality")
    axes[2].legend(loc="upper left")
    _format_axes(axes[2])

    # 4. Residual
    axes[3].plot(t_train, resid, label="Residual (what MLP learns)")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Months")
    axes[3].legend(loc="upper left")
    _format_axes(axes[3])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_path:
        p = Path(save_path)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Saved STL decomposition figure to {p}")

    return fig

from load_data import load_monthly_finance_data

# Load one series from the M3 financial dataset
data = load_monthly_finance_data()
series = data[0]["train"]
h = len(data[0]["test"])

# Plot STL decomposition 
plot_stl_decomposition_example(series, h=h, season=12, save_path="fig_stl_example.png")

######################################################################################################

from typing import Optional, Union
from baselines import seasonal_naive_forecast


def plot_forecast_example(
    train: np.ndarray,
    test: np.ndarray,
    mlp_level_forecast: np.ndarray,   
    season: int = 12,
) -> Path:
    """
    Plot forecast comparison for one series:
    Observed vs Seasonal Naïve vs MLP.
    """
    _apply_paper_style()

    h = len(test)

 
    snaive_level = seasonal_naive_forecast(train, h=h, season=season)

    t_train = np.arange(len(train))
    t_test = np.arange(len(train), len(train) + h)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

  
    ax.axvspan(len(train) - 1, len(train) + h, color="0.9", alpha=0.4, label="Forecast Horizon")
    ax.plot(t_train, train, color="0.6", label="Training Data")
    ax.plot(t_test, test, color="black", linewidth=1.5, label="Observed (Test)")
    ax.plot(t_test, snaive_level, "--", linewidth=1.2, label="Seasonal Naïve")
    ax.plot(t_test, mlp_level_forecast, ":", linewidth=1.2, label="MLP Forecast")
    ax.set_xlabel("Months")
    ax.set_ylabel("Index value")  
    ax.set_title("Forecast Example: Observed vs. Baselines")
    ax.legend(loc="lower left")  
    _format_axes(ax)

    plt.tight_layout()

    project_root = Path(__file__).resolve().parents[1]  
    outdir = project_root / "figures"
    outdir.mkdir(exist_ok=True)
    base = outdir / "fig_forecast_example"

    plt.savefig(f"{base}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved forecast comparison figure:\n  - {base}.png")
    return base


# Run ONE REAL SERIES end-to-end
if __name__ == "__main__":
    from load_data import load_monthly_finance_data
    from train_model import cv_select_hyperparams, train_one_series, forecast_series

    data = load_monthly_finance_data()  
    assert len(data) > 0, "No series loaded."

    # Pick which series to plot 
    idx = 0
    item = data[idx]
    y_tr, y_te = item["train"], item["test"]
    h = len(y_te)
    season = 12

    print(f"Running single-series experiment for index {idx} (train={len(y_tr)}, test={h})")

    # rolling CV
    cfg = cv_select_hyperparams(y_tr, h)
    print(f"Selected hyperparameters: {cfg}")

    # Train final model with early stopping
    model, comps, hist = train_one_series(
        y_tr, h,
        input_len=cfg["input_len"],
        hidden=cfg["hidden"],
        lr=cfg["lr"],
        wd=cfg["wd"],
        epochs=180,
        patience=8,
    )
    assert model is not None, "Training failed or returned None model."

    yhat_test_level = forecast_series(model, comps, y_tr, h)
    plot_forecast_example(y_tr, y_te, yhat_test_level, season=season)

################################################################################################
#Scatterplot and boxplot

def plot_smape_comparison(
    results_csv: str = "figures/results_finance_monthly.csv",
    save_dir: str = "figures"
):
    """
    Creates separate plots comparing per-series sMAPE between the MLP model
    and the Seasonal Naïve benchmark:
      1) Boxplot of sMAPE distributions
      2) Scatter plot of per-series sMAPE
    """

    df = pd.read_csv(results_csv)
    required_cols = ["test_smape", "snaive_smape"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {results_csv}")

    smape_model = df["test_smape"]
    smape_snaive = df["snaive_smape"]

    save_path_box = Path(save_dir) / "smape_boxplot.png"
    save_path_scatter = Path(save_dir) / "smape_scatter.png"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # --- Boxplot ---
    plt.figure(figsize=(5, 4))
    plt.boxplot([smape_model, smape_snaive], labels=["MLP", "Seasonal Naïve"])
    plt.title("Distribution of sMAPE Across All Series")
    plt.ylabel("sMAPE (%)")
    plt.tight_layout()
    plt.savefig(save_path_box, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved boxplot: {save_path_box}")

    # --- Scatter plot ---
    plt.figure(figsize=(5, 4))
    plt.scatter(smape_snaive, smape_model, alpha=0.6)
    plt.plot([smape_snaive.min(), smape_snaive.max()],
             [smape_snaive.min(), smape_snaive.max()],
             "k--", linewidth=1, label="Equal performance (y = x)")
    plt.xlabel("Seasonal Naïve sMAPE (%)")
    plt.ylabel("MLP sMAPE (%)")
    plt.title("sMAPE: MLP vs. Seasonal Naïve")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path_scatter, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved scatter plot: {save_path_scatter}")


if __name__ == "__main__":
    plot_smape_comparison()

