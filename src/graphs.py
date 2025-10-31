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
