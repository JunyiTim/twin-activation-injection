"""Paired re-analysis of the affective-priming injection experiment.

Compares the original independent-samples Welch analysis with a paired analysis
(every persona appears in every condition), on the two final generation logs.
Usage: python paired_reanalysis.py [gratitude_log.csv] [control_log.csv]
"""
import ast
import sys

import numpy as np
import pandas as pd
from scipy import stats

GRAT = sys.argv[1] if len(sys.argv) > 1 else "data/gratitude_generation_log.csv.gz"
CTRL = sys.argv[2] if len(sys.argv) > 2 else "data/lackofcontrol_generation_log.csv.gz"
OUT = "paired_reanalysis_results.csv"

MEASURES = ["manip_check", "elevation", "dv1"]
LABELS = {1.8: "Low", 2.5: "Moderate", 3.2: "High", 3.9: "Extreme"}
DOSE_ARMS = {"inject_gratitude", "inject_lackofcontrol"}
ACTIVE_ARMS = {"inject_routine", "inject_incontrol"}


# Reverse-coded items (1-based) as in the original study's scoring code
# (mega_study_evaluation/affective_priming/mega_study_evaluation.py): 8 - x, then average.
REVERSE = {"control": {"elevation": [3, 7, 8],   # Fatigue scale items 3, 7, 8
                       "dv1": [1, 4, 5]}}        # Desire for Predictability items 1, 4, 5


def mean_of_list(x, reverse_items=()):
    try:
        v = ast.literal_eval(x) if isinstance(x, str) else x
        if not (isinstance(v, list) and len(v) > 0):
            return np.nan
        v = [float(t) for t in v]
        for i in reverse_items:
            v[i - 1] = 8.0 - v[i - 1]
        return float(np.mean(v))
    except (ValueError, SyntaxError):
        return np.nan


def condition(row):
    if row["arm"] in ("real_prime", "real_control"):
        return row["arm"]
    b = round(float(row["beta"]), 1)
    if row["arm"] in DOSE_ARMS:
        return f"dose_{LABELS[b]}"
    if row["arm"] in ACTIVE_ARMS:
        return "noinj" if b == 0.0 else "active_Moderate"
    return None


def wide(path, domain=None):
    df = pd.read_csv(path)
    if domain is None:
        domain = "control" if "inject_incontrol" in set(df["arm"].dropna()) else "gratitude"
    df = df[(df["is_final"] == True) & (df["success"] == True)].copy()
    df["cond"] = df.apply(condition, axis=1)
    for m in MEASURES:
        rev = REVERSE.get(domain, {}).get(m, ())
        df[m] = df[m].apply(lambda x: mean_of_list(x, rev))
    w = df.pivot(index="pid", columns="cond", values=MEASURES)
    return w


def cohen_d(x, y):
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / sp if sp > 0 else np.nan


def compare(w, a, b, m):
    x, y = w[(m, a)], w[(m, b)]
    xa, yb = x.dropna(), y.dropna()
    tw, pw = stats.ttest_ind(xa, yb, equal_var=False)
    both = x.notna() & y.notna()
    diff = (x - y)[both]
    tp, pp = stats.ttest_rel(x[both], y[both])
    dz = diff.mean() / diff.std(ddof=1)
    r = np.corrcoef(x[both], y[both])[0, 1]
    return dict(a=a, b=b, measure=m, n_a=len(xa), n_b=len(yb), n_pairs=int(both.sum()),
                diff=diff.mean(), d_welch=cohen_d(xa, yb), t_welch=tw, p_welch=pw,
                dz=dz, t_paired=tp, p_paired=pp, r=r)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def full_grid(w, domain):
    conds = [c for c in w.columns.get_level_values(1).unique()]
    inj = [c for c in conds if c not in ("real_prime", "real_control")]
    rows = []
    for m in MEASURES:
        rows.append(compare(w, "real_prime", "real_control", m))
        for c in inj:
            for real in ("real_prime", "real_control"):
                rows.append(compare(w, c, real, m))
    out = pd.DataFrame(rows)
    out.insert(0, "domain", domain)
    out["p_paired_holm"] = holm(out["p_paired"].values)
    return out


# key comparisons reported in the manuscript (Tables 8 and 9)
KEY = {
    "gratitude": [("dose_Low", "real_prime", "manip_check"), ("dose_Extreme", "real_prime", "manip_check"),
                  ("dose_Low", "real_prime", "elevation"), ("real_prime", "real_control", "elevation"),
                  ("real_prime", "real_control", "dv1"), ("active_Moderate", "real_prime", "dv1"),
                  ("active_Moderate", "real_control", "dv1"), ("noinj", "real_control", "elevation")],
    "control": [("dose_Low", "real_prime", "manip_check"), ("active_Moderate", "real_control", "manip_check"),
                ("real_prime", "real_control", "elevation"), ("real_prime", "real_control", "dv1"),
                ("active_Moderate", "real_prime", "elevation"), ("active_Moderate", "real_control", "elevation"),
                ("active_Moderate", "real_prime", "dv1"), ("active_Moderate", "real_control", "dv1")],
}

if __name__ == "__main__":
    grids = []
    for domain, path in (("gratitude", GRAT), ("control", CTRL)):
        w = wide(path)
        g = full_grid(w, domain)
        grids.append(g)
    allg = pd.concat(grids, ignore_index=True)
    allg.to_csv(OUT, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for domain in ("gratitude", "control"):
        g = allg[allg.domain == domain]
        rows = []
        for a, b, m in KEY[domain]:
            r = g[(g.a == a) & (g.b == b) & (g.measure == m)].iloc[0]
            rows.append(r)
        k = pd.DataFrame(rows)[["a", "b", "measure", "n_pairs", "d_welch", "t_welch", "p_welch",
                                 "dz", "t_paired", "p_paired", "p_paired_holm", "r"]]
        print(f"\n=== {domain}: key comparisons (diff = first minus second) ===")
        print(k.round(4).to_string(index=False))
    print(f"\nfull grid saved to {OUT} ({len(allg)} comparisons)")
