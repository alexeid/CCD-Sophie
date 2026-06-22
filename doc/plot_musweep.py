#!/usr/bin/env python3
"""Plot the KRegCCD mu-calibration sweep as three "lower is better" loss curves that all bend the
same way and share a near-coincident minimum:

  (1) held-out surprisal   -mean logP/tree on the SECOND-half held-out trees,
  (2) CV surprisal         -mean logP/tree from leave-k-out cross-validation within the FIRST half,
  (3) novelty miscalibration  |P(novel) - empirical novel-clade rate|.

Each is min-max normalised to [0, 1] (0 at its own optimum) so the three can overlay on one axis;
a dotted vertical marks each curve's optimal mu.

Reads the TSV written by ccd.experiments.regularisation.RSV2ThreeModelTable
(columns: mu, pnovel, empirical, meanLogP, cvLogP).

Usage: plot_musweep.py <out.pdf> <sweep.tsv> [--title="..."]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1E5AC5"   # held-out (second half)
GREEN = "#1B9E77"  # cross-validation (training half)
RED = "#C51E3A"    # novelty miscalibration


def norm01(y):
    """Min-max normalise to [0, 1]; flat arrays map to 0."""
    y = np.asarray(y, float)
    lo, hi = np.nanmin(y), np.nanmax(y)
    return np.zeros_like(y) if hi <= lo else (y - lo) / (hi - lo)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    title = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--title=")), None)
    if len(args) < 2:
        sys.exit('usage: plot_musweep.py <out.pdf> <sweep.tsv> [--title="..."]')
    out, tsv = args[0], args[1]

    d = np.genfromtxt(tsv, delimiter="\t", names=True)
    mu = d["mu"]
    emp = float(d["empirical"][0])

    # Three losses, all "lower is better".
    held = -d["meanLogP"]                       # held-out surprisal (second half)
    cv = -d["cvLogP"]                           # CV surprisal (training half)
    miscal = np.abs(d["pnovel"] - emp)          # |P(novel) - empirical rate|

    curves = [
        (held, BLUE, "held-out surprisal  ($-\\log P$/tree, 2nd half)"),
        (cv, GREEN, "CV surprisal  ($-\\log P$/tree, leave-$k$-out)"),
        (miscal, RED, "novelty miscalibration  $|P(\\mathrm{novel})-\\hat p|$"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    handles = []
    for y, color, label in curves:
        yn = norm01(y)
        (line,) = ax.plot(mu, yn, color=color, lw=2.2, label=label)
        handles.append(line)
        muStar = mu[int(np.nanargmin(y))]
        ax.axvline(muStar, color=color, ls=":", lw=1.3)
        ax.annotate(f"$\\mu$ = {muStar:.4f}", xy=(muStar, 0.0),
                    xytext=(4, 18), textcoords="offset points", rotation=90,
                    fontsize=7.5, color=color, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(mu.min(), mu.max())
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("escape probability  $\\mu$")
    ax.set_ylabel("relative loss  (0 = each curve's optimum)")
    ax.legend(handles=handles, loc="upper center", fontsize=8.2, framealpha=0.9)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else (0, 0, 1, 1))
    fig.savefig(out)

    opts = {lbl.split("  ")[0]: mu[int(np.nanargmin(y))] for y, _, lbl in curves}
    print(f"wrote {out}  optima: " + ", ".join(f"{k}={v:.5f}" for k, v in opts.items())
          + f"  (empirical rate={emp:.3f})")


if __name__ == "__main__":
    main()
