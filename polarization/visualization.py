"""
Visualization helpers for polarization / consensus analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .weights import weight_matrix_generator
from .scores import calculate_scores


# ---------------------------------------------------------------------------
# Weight matrix heatmap
# ---------------------------------------------------------------------------

def weight_matrix_visualization(
    start_value: int = 0,
    end_value: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    type: str = "polarization",
    kernel: str = "power",
) -> None:
    """
    Display a heatmap of the weight matrix produced by
    :func:`~polarization.weights.weight_matrix_generator`.

    Parameters
    ----------
    start_value, end_value : int
        Scale bounds.
    p, q : float
        Kernel / agreement parameters.
    type : {'polarization', 'consensus'}
    kernel : {'power', 'gaussian'}
    """
    N = end_value - start_value
    weights = weight_matrix_generator(
        start_value=start_value, end_value=end_value,
        p=p, q=q, type=type, kernel=kernel,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(weights, cmap="coolwarm", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    scale_labels = list(range(start_value, end_value + 1))
    ax.set_xticks(range(N + 1))
    ax.set_xticklabels(scale_labels)
    ax.set_yticks(range(N + 1))
    ax.set_yticklabels(scale_labels)
    ax.set_title(
        f"{type.capitalize()} Weight Matrix — kernel={kernel} (p={p}, q={q})"
    )

    for i in range(N + 1):
        for j in range(N + 1):
            value = weights[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color=text_color, fontsize=9)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Full dashboard
# ---------------------------------------------------------------------------

def dashboard_pair(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    score_type: str = "polarization",
    start_value: int = 0,
    end_value: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    kernel: str = "power",
    dropna: bool = False,
    B_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute scores and display a three-panel dashboard for variable pair (*x*, *y*).

    Panels
    ------
    A — Bootstrap confidence intervals (horizontal bar chart).
    B — Bootstrap distributions (overlaid histograms).
    C — Per-cell WP heatmap with the main or anti-diagonal separator.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str
        Column names.
    score_type : {'polarization', 'consensus'}
    start_value, end_value : int
    p, q : float
    kernel : {'power', 'gaussian'}
    dropna : bool
    B_boot : int
        Bootstrap replications for this dashboard run.
    ci : float
        Confidence level.
    random_state : int

    Returns
    -------
    dict
        ``{'scores': <calculate_scores result>}``
    """
    res = calculate_scores(
        df, x, y,
        start_value=start_value, end_value=end_value, p=p, q=q,
        score_type=score_type, kernel=kernel, dropna=dropna,
        B=B_boot, ci=ci, random_state=random_state,
        keep_dists=True,
    )

    point = res["point"]
    boot  = res["bootstrap"]

    if score_type == "polarization":
        keys = [k for k in boot if k != "overall"]
    else:
        keys = list(boot.keys())

    # ── Print summary ─────────────────────────────────────────────────────
    sep = "=" * 54
    print(f"\n{sep}")
    print(f"  {score_type.upper()} DASHBOARD  —  {x}  ×  {y}")
    print(sep)
    print(f"  kernel={kernel}  |  B={B_boot}  |  CI={int(ci * 100)}%")
    print(f"  Diagnostics: sparsity={point['_sparsity']:.2%}")
    print("-" * 54)
    print("  Bootstrap results:")
    for k in keys:
        ci_k = boot[k]["ci"]
        print(
            f"    {k:>10s}: {point[k]:.4f}  "
            f"[{ci_k[0]:.4f}, {ci_k[1]:.4f}]  SE={boot[k]['se']:.4f}"
        )
    print(f"{sep}\n")

    wp     = point["_wp_matrix"]
    labels = point["_x_labels"]
    n      = len(labels)
    I_idx, J_idx = np.indices((n, n))

    if score_type == "polarization":
        col_contrib_y = np.array([wp[:j, j].sum() for j in range(n)])

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 18))
    gs  = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    ax_b  = fig.add_subplot(gs[0, 0])
    ax_pk = fig.add_subplot(gs[0, 1])
    ax_hm = fig.add_subplot(gs[1:, :])

    fig.suptitle(
        f"{score_type.title()} Dashboard  [{kernel}]  —  {x}  ×  {y}",
        fontsize=13, fontweight="bold",
    )

    # ── Panel A: Bootstrap CIs ────────────────────────────────────────────
    y_pos      = np.arange(len(keys))
    point_vals = np.array([point[k] for k in keys], dtype=float)
    ci_lo      = np.array([boot[k]["ci"][0] for k in keys], dtype=float)
    ci_hi      = np.array([boot[k]["ci"][1] for k in keys], dtype=float)
    colors     = ["#4878CF" if v >= 0 else "#D65F5F" for v in point_vals]

    ax_b.barh(y_pos, point_vals, color=colors, alpha=0.75, height=0.5)
    ax_b.errorbar(
        point_vals, y_pos,
        xerr=np.vstack([point_vals - ci_lo, ci_hi - point_vals]),
        fmt="none", color="black", capsize=4, linewidth=1.2,
    )
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(keys, fontsize=9)
    ax_b.set_xlabel("Score", fontsize=9)
    ax_b.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_b.set_title(f"Bootstrap Results\n(B={B_boot}, {int(ci * 100)}% CI)", fontsize=10)
    for i, (k, v) in enumerate(zip(keys, point_vals)):
        offset = 0.005 if v >= 0 else -0.005
        ax_b.text(v + offset, i, f"{v:.3f}", va="center",
                  ha="left" if v >= 0 else "right", fontsize=8)

    # ── Panel B: Bootstrap distributions ─────────────────────────────────
    palette = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66"]
    for i, k in enumerate(keys):
        color = palette[i % len(palette)]
        ax_pk.hist(boot[k]["dist"], bins=35, alpha=0.55, color=color,
                   edgecolor="white", label=f"{k} ({point[k]:.3f})")
        ax_pk.axvline(point[k], color=color, linestyle="--", linewidth=1.5)
    ax_pk.set_xlabel("Score (bootstrap)", fontsize=9)
    ax_pk.set_ylabel("Count", fontsize=9)
    ax_pk.set_title("Bootstrap Distributions\n(all keys)", fontsize=10)
    ax_pk.legend(fontsize=7)

    # ── Panel C: WP heatmap ───────────────────────────────────────────────
    vmax = np.abs(wp).max() if np.abs(wp).max() > 0 else 1
    im = ax_hm.imshow(wp, cmap="RdBu", origin="upper", aspect="auto",
                      vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.02)

    ax_hm.set_xticks(range(n))
    ax_hm.set_xticklabels(labels, fontsize=10)
    ax_hm.set_yticks(range(n))
    ax_hm.set_yticklabels(labels, fontsize=10)
    ax_hm.set_xlabel(y, fontsize=11)
    ax_hm.set_ylabel(x, fontsize=11)
    ax_hm.set_title(
        f"Per-cell {score_type.title()} Scores  (W × P matrix)", fontsize=12
    )

    thresh = vmax * 0.4
    for i in range(n):
        for j in range(n):
            v = wp[i, j]
            if v != 0:
                ax_hm.text(j, i, f"{v:.3f}", ha="center", va="center",
                           fontsize=7,
                           color="white" if abs(v) > thresh else "black")

    if score_type == "polarization":
        ax_hm.plot([-0.5, n - 0.5], [-0.5, n - 0.5],
                   color="black", linewidth=2, linestyle="--", alpha=0.6)
        ax3 = ax_hm.twiny()
        ax3.set_xlim(ax_hm.get_xlim())

    else:
        ax_hm.plot([-0.5, n - 0.5], [n - 0.5, -0.5],
                   color="black", linewidth=2, linestyle="--", alpha=0.6)

    plt.show()
    return {"scores": res}
