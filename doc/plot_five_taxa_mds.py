#!/usr/bin/env python3
"""MDS of Robinson-Foulds distances between the 105 rooted 5-taxon topologies, with marker area
proportional to each tree's KRegCCD probability and colour by novel-clade count.

Reads doc/kreg-five-taxa.tsv (columns: newick, prob, novel) written by
test.ccd.model.KRegFiveTaxaTest. Rooted RF = size of the symmetric difference of the two trees'
non-trivial clade sets. Layout by classical (Torgerson) MDS.

Usage: plot_five_taxa_mds.py <out.pdf> <tsv> [--title="..."]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def clades(nwk):
    """All clades (frozensets of leaf names) of a names-only newick like (E,(D,(C,(B,A))))."""
    cl = set()

    def rec(s):
        if "(" not in s:
            f = frozenset([s])
            cl.add(f)
            return f
        inner = s[1:-1]
        depth, cur, parts = 0, "", []
        for ch in inner:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur); cur = ""
            else:
                cur += ch
        parts.append(cur)
        f = frozenset().union(*[rec(p) for p in parts])
        cl.add(f)
        return f

    rec(nwk.strip().rstrip(";"))
    return cl


def nontrivial(cl, n):
    return {c for c in cl if 2 <= len(c) <= n - 1}


def classical_mds(D, dims=2):
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:dims]
    L = np.sqrt(np.clip(vals[idx], 0, None))
    return vecs[:, idx] * L


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    title = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--title=")), None)
    if len(pos) < 2:
        sys.exit('usage: plot_five_taxa_mds.py <out.pdf> <tsv> [--title="..."]')
    out, tsv = pos[0], pos[1]

    nwks, probs, novel = [], [], []
    with open(tsv) as fh:
        next(fh)
        for line in fh:
            a, p, nv = line.rstrip("\n").split("\t")
            nwks.append(a); probs.append(float(p)); novel.append(int(nv))
    probs = np.array(probs); novel = np.array(novel)
    n = len(nwks)
    ntax = max(len(c) for c in clades(nwks[0]))

    cl = [nontrivial(clades(w), ntax) for w in nwks]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = len(cl[i] ^ cl[j])

    xy = classical_mds(D, 2)
    # tiny deterministic jitter so RF-coincident points don't fully overlap
    rng = np.random.RandomState(0)
    xy = xy + rng.normal(0, 0.05, xy.shape)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    cmap = matplotlib.colors.ListedColormap(["#C51E3A", "#1E5AC5", "#1B9E77", "#7B3FA0"])
    # marker AREA proportional to probability (floor so tiny tiers stay visible)
    s = probs / probs.max() * 2600 + 8
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=novel, cmap=cmap, vmin=-0.5, vmax=3.5,
                    alpha=0.78, edgecolors="white", linewidths=0.5, zorder=2)

    # annotate the backbone trees (2 observed + the alpha-expansion)
    order = np.argsort(probs)[::-1]
    labels = {order[0]: "$T_1$", order[1]: "$T_2$", order[2]: "expansion"}
    for i, lab in labels.items():
        ax.annotate(lab, xy=(xy[i, 0], xy[i, 1]), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, fontweight="bold")

    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2")
    ax.set_aspect("equal", adjustable="datalim")
    handles = [Line2D([0], [0], marker="o", ls="", mfc=cmap(k), mec="white",
                      ms=8, label=f"{k} novel clade" + ("s" if k != 1 else ""))
               for k in range(4)]
    leg1 = ax.legend(handles=handles, title="topology", loc="upper left", fontsize=8, title_fontsize=8.5)
    ax.add_artist(leg1)
    # size legend
    for p, lab in [(probs.max(), "0.375"), (0.0234, "0.023"), (5.4e-7, "5e-7")]:
        ax.scatter([], [], s=p / probs.max() * 2600 + 8, c="0.6", edgecolors="white",
                   linewidths=0.5, label=lab)
    ax.legend(loc="lower right", fontsize=8, title="P(tree)  (area)", title_fontsize=8.5,
              labelspacing=1.6, borderpad=1.0, scatterpoints=1)
    ax.set_title(title or "105 rooted 5-taxon trees: MDS of RF distance, area $\\propto P$")
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}  ({n} trees, RF in [{int(D.min())},{int(D.max())}], "
          f"{len(set(np.round(probs,12)))} probability tiers)")


if __name__ == "__main__":
    main()
