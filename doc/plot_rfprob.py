#!/usr/bin/env python3
"""Average KRegCCD probability (with spread) as a function of mean rooted-RF distance to the
posterior sample. Works for any taxon count.

For each enumerated tree, dbar(T) = mean rooted-RF distance to the sampled trees (default the two
5-taxon trees; pass --t1/--t2 for other samples). Trees are grouped by dbar; we report per group
the count, mean P, sd P, min/max P, and total mass, and plot mean P (log y) with min-max whiskers
and marker size by group count.

Reads a TSV with columns: newick, prob, novel.
Usage: plot_rfprob.py <out.pdf> <tsv> [--t1=NWK --t2=NWK] [--title="..."]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def clades(nwk):
    cl = set()

    def rec(s):
        if "(" not in s:
            f = frozenset([s]); cl.add(f); return f
        depth, cur, parts = 0, "", []
        for ch in s[1:-1]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur); cur = ""
            else:
                cur += ch
        parts.append(cur)
        f = frozenset().union(*[rec(p) for p in parts]); cl.add(f); return f

    rec(nwk.strip().rstrip(";"))
    return cl


def nontrivial(nwk, n):
    return {c for c in clades(nwk) if 2 <= len(c) <= n - 1}


def opt(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(name + "=")), default)


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    out, tsv = pos[0], pos[1]
    t1 = opt("--t1", "((((A,B),C),D),E)")
    t2 = opt("--t2", "((((D,C),B),A),E)")
    title = opt("--title", None)

    nwks, probs = [], []
    with open(tsv) as fh:
        next(fh)
        for line in fh:
            a, p, _ = line.rstrip("\n").split("\t")
            nwks.append(a); probs.append(float(p))
    probs = np.array(probs)
    n = max(len(c) for c in clades(nwks[0]))

    S = [nontrivial(t1, n), nontrivial(t2, n)]
    cl = [nontrivial(w, n) for w in nwks]
    dbar = np.array([np.mean([len(c ^ s) for s in S]) for c in cl])

    print(f"{'mean RF':>8} {'count':>6} {'mean P':>11} {'sd P':>11} {'min P':>11} {'max P':>11} {'mass':>10}")
    ds, mp, sp, lo, hi, cnt = [], [], [], [], [], []
    for d in sorted(set(dbar)):
        v = probs[dbar == d]
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"{d:8.1f} {len(v):6d} {v.mean():11.3e} {sd:11.3e} {v.min():11.3e} {v.max():11.3e} {v.sum():10.6f}")
        ds.append(d); mp.append(v.mean()); sp.append(sd); lo.append(v.min()); hi.append(v.max()); cnt.append(len(v))
    ds, mp, lo, hi, cnt = map(np.array, (ds, mp, lo, hi, cnt))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(ds, mp, "-", color="0.7", lw=1.2, zorder=1)
    # min-max whisker per tier (honest on a log axis; sd is in the printed table)
    ax.vlines(ds, lo, hi, color="#1E5AC5", lw=1.0, alpha=0.5, zorder=1)
    ax.scatter(ds, mp, s=np.sqrt(cnt) * 70 + 20, c="#1E5AC5", edgecolors="white", linewidths=0.6, zorder=3)
    for d, p, c in zip(ds, hi, cnt):
        ax.annotate(f"n={c}", xy=(d, p), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7.5, color="0.3")
    ax.set_yscale("log")
    ax.set_xlabel("mean rooted-RF distance to posterior sample")
    ax.set_ylabel("$P(\\mathrm{tree})$:  mean (point), min–max (bar), log scale")
    ax.set_title(title or "KRegCCD probability vs RF distance from the sample")
    fig.tight_layout()
    fig.savefig(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
