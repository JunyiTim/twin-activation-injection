"""Builds results_revised.tex, discussion_revised.tex, abstract_revised.tex and appendix_grid.tex
from paired_reanalysis_results.csv (so every number in the text and tables comes from the data).
Run paired_reanalysis.py first.  Usage: python build_results.py
"""
import re

import numpy as np
import pandas as pd

import paired_reanalysis as pr

G = pd.read_csv("paired_reanalysis_results.csv")

# within-path contrasts: injection condition minus No-Injection (same prompt/output structure)
_wrows = []
for _dom, _path in (("gratitude", pr.GRAT), ("control", pr.CTRL)):
    _w = pr.wide(_path)
    _rows = []
    for _c in ["dose_Low", "dose_Moderate", "dose_High", "dose_Extreme", "active_Moderate"]:
        for _m in pr.MEASURES:
            _rows.append(pr.compare(_w, _c, "noinj", _m))
    _d = pd.DataFrame(_rows)
    _d["p_paired_holm"] = pr.holm(_d.p_paired.values)
    _d.insert(0, "domain", _dom)
    _wrows.append(_d)
WD = pd.concat(_wrows, ignore_index=True)
WD.to_csv("within_path_results.csv", index=False)
LEVEL = {1.8: "Low", 2.5: "Moderate", 3.2: "High", 3.9: "Extreme"}
NAME_FULL = {"real_prime": "Prime", "real_control": "Control", "noinj": "No-Injection",
             "active_Moderate": "Active-Control Injection (Moderate)"}
for b, l in LEVEL.items():
    NAME_FULL[f"dose_{l}"] = f"Dose Injection ({l})"
NAME_SHORT = {k: v.replace("Active-Control Injection", "Active-Ctrl").replace("Dose Injection", "Dose")
              for k, v in NAME_FULL.items()}


def row(dom, a, b, m):
    r = G[(G.domain == dom) & (G.a == a) & (G.b == b) & (G.measure == m)]
    assert len(r) == 1, (dom, a, b, m)
    return r.iloc[0]


def pstr(p, eq="="):
    if p < 0.0001:
        return "p<.0001"
    s = f"{p:.4f}" if p < 0.01 else f"{p:.3f}"
    return f"p{eq}" + s.lstrip("0")


def pbare(p):
    return "<.0001" if p < 0.0001 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}").lstrip("0")


def num(x, nd=2):
    return f"{x:.{nd}f}"


def stat(dom, a, b, m, kind):
    r = row(dom, a, b, m)
    df = int(r.n_pairs) - 1
    if kind == "full":
        return f"$t({df})={num(r.t_paired)}$, ${pstr(r.p_paired)}$, $d_z={num(r.dz)}$"
    if kind == "tp":
        return f"$t({df})={num(r.t_paired)}$, ${pstr(r.p_paired)}$"
    if kind == "dz":
        return f"$d_z={num(r.dz)}$"
    if kind == "dzp":
        return f"$d_z={num(r.dz)}$, ${pstr(r.p_paired)}$"
    if kind == "holm":
        return f"${pstr(r.p_paired_holm)}$"
    if kind == "diff":
        return f"${num(r['diff'])}$"
    if kind == "r":
        return num(r.r)
    if kind == "p":
        return f"${pstr(r.p_paired)}$"
    raise ValueError(kind)


def wrow(dom, cond, m):
    r = WD[(WD.domain == dom) & (WD.a == cond) & (WD.measure == m)]
    assert len(r) == 1, (dom, cond, m)
    return r.iloc[0]


def wstat(dom, cond, m, kind):
    r = wrow(dom, cond, m)
    df = int(r.n_pairs) - 1
    if kind == "full":
        return f"$t({df})={num(r.t_paired)}$, ${pstr(r.p_paired)}$, $d_z={num(r.dz)}$"
    if kind == "tp":
        return f"$t({df})={num(r.t_paired)}$, ${pstr(r.p_paired)}$"
    if kind == "dz":
        return f"$d_z={num(r.dz)}$"
    if kind == "holm":
        return f"${pstr(r.p_paired_holm)}$"
    if kind == "diff":
        return f"${num(r['diff'])}$"
    raise ValueError(kind)


# ---- bootstrap CI for the matching strength (gratitude dose curve vs real prime, manipulation check)
def matching_strength(sub, measure="manip_check"):
    betas = [1.8, 2.5, 3.2, 3.9]
    m = [sub[(measure, f"dose_{LEVEL[b]}")].mean() for b in betas]
    t = sub[(measure, "real_prime")].mean()
    for i in range(3):
        if (m[i] - t) * (m[i + 1] - t) <= 0 and m[i] != m[i + 1]:
            return betas[i] + (t - m[i]) / (m[i + 1] - m[i]) * (betas[i + 1] - betas[i])
    return np.nan


wg = pr.wide(pr.GRAT)
rng = np.random.default_rng(0)
est = matching_strength(wg)
bs = np.array([matching_strength(wg.iloc[rng.integers(0, len(wg), len(wg))]) for _ in range(5000)])
ok = bs[~np.isnan(bs)]
BOOT = dict(est=f"{est:.1f}", lo=f"{np.percentile(ok, 2.5):.1f}", hi=f"{np.percentile(ok, 97.5):.1f}",
            share=f"{100 * len(ok) / len(bs):.0f}")
print("bootstrap matching strength:", BOOT)


# ---------------- table builders ----------------
def key_table(spec, caption, label, width_cmp="4.4cm"):
    out = [r"\begin{table}[htbp]", r"\centering", r"\renewcommand{\arraystretch}{1.25}",
           r"\begin{tabular}{@{} >{\raggedright\arraybackslash}p{1.4cm} >{\raggedright\arraybackslash}p{" + width_cmp
           + r"} l c c c @{}}", r"\toprule", r"\rowcolor{axisblue}",
           r"\textbf{Domain} & \textbf{Comparison} & \textbf{Measure} & \textbf{$d_z$} & \textbf{$t$ ($p$)} & \textbf{$p_{\text{Holm}}$} \\",
           r"\midrule"]
    for gi, (dom, label_dom, items) in enumerate(spec):
        if gi > 0:
            out.append(r"\midrule")
        for i, (a, b, m) in enumerate(items):
            r = row(dom, a, b, m)
            mname = {"manip_check": r"manip.\ check", "elevation": "elevation" if dom == "gratitude" else "fatigue",
                     "dv1": "dv1"}[m]
            first = r"\multirow{" + str(len(items)) + r"}{1.4cm}{" + label_dom + "}" if i == 0 else ""
            cmp_ = f"{NAME_FULL[a]} vs.\\ {NAME_FULL[b]}"
            out.append(f"  {first} & {cmp_} & {mname} & ${num(r.dz)}$ & ${num(r.t_paired)}$ (${pstr(r.p_paired)}$) & {pbare(r.p_paired_holm)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(out)


def within_table(spec, caption, label):
    out = [r"\begin{table}[htbp]", r"\centering", r"\renewcommand{\arraystretch}{1.25}",
           r"\begin{tabular}{@{} >{\raggedright\arraybackslash}p{1.4cm} >{\raggedright\arraybackslash}p{5.0cm} l c c c @{}}",
           r"\toprule", r"\rowcolor{axisblue}",
           r"\textbf{Domain} & \textbf{Comparison} & \textbf{Measure} & \textbf{$d_z$} & \textbf{$t$ ($p$)} & \textbf{$p_{\text{Holm}}$} \\",
           r"\midrule"]
    for gi, (dom, label_dom, items) in enumerate(spec):
        if gi > 0:
            out.append(r"\midrule")
        for i, (cond, m) in enumerate(items):
            r = wrow(dom, cond, m)
            mname = {"manip_check": r"manip.\ check", "elevation": "proximal" if dom == "gratitude" else "proximal (fatigue)",
                     "dv1": "distal"}[m]
            first = r"\multirow{" + str(len(items)) + r"}{1.4cm}{" + label_dom + "}" if i == 0 else ""
            out.append(f"  {first} & {NAME_FULL[cond]} vs.\\ No-Injection & {mname} & ${num(r.dz)}$ & ${num(r.t_paired)}$ (${pstr(r.p_paired)}$) & {pbare(r.p_paired_holm)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(out)


TAB_WITHIN = within_table(
    [("gratitude", "Gratitude", [("active_Moderate", "manip_check"), ("active_Moderate", "elevation"), ("active_Moderate", "dv1")]),
     ("control", "Lack of Control", [("active_Moderate", "manip_check"), ("active_Moderate", "elevation"), ("active_Moderate", "dv1")])],
    r"Within-path paired comparisons: active-control arm (Moderate) minus No-Injection. Both conditions share the same prompt and output structure, so differences reflect the injected vector alone. $p_{\text{Holm}}$ is adjusted over the 15 within-path comparisons per domain in Appendix~\ref{app:grid}.",
    "tab:within-path")


wc = pr.wide(pr.CTRL)
PUBLISHED = {  # (human effect, original digital-twin effect), prime minus control; from the original study's Figures 24 and 26
    ("gratitude", "manip_check"): (0.81, 1.33), ("gratitude", "elevation"): (0.79, 0.77), ("gratitude", "dv1"): (-0.01, 0.24),
    ("control", "manip_check"): (0.31, 1.00), ("control", "elevation"): (0.18, 0.88), ("control", "dv1"): (0.25, 0.05)}
MLAB = {("gratitude", "manip_check"): "Manipulation check", ("gratitude", "elevation"): "Proximal: Elevation",
        ("gratitude", "dv1"): "Distal: Empathic Concern", ("control", "manip_check"): "Manipulation check",
        ("control", "elevation"): "Proximal: Fatigue", ("control", "dv1"): "Distal: Desire for Predictability"}


def our_effect(dom, m):
    w = wg if dom == "gratitude" else wc
    pm, cm = w[(m, "real_prime")].mean(), w[(m, "real_control")].mean()
    return (cm - pm) if (dom == "control" and m == "manip_check") else (pm - cm)


CAL = []
for (dom, m), (h, t) in PUBLISHED.items():
    e = our_effect(dom, m)
    CAL.append((dom, m, h, e, t, min(h, t) - 1e-9 <= e <= max(h, t) + 1e-9))
N_BETWEEN = sum(1 for c in CAL if c[5])
EXC = [c for c in CAL if not c[5]]
if EXC:
    parts_ = []
    for dom, m, h, e, t, _ in EXC:
        side = "smaller" if e < min(h, t) else "larger"
        parts_.append(f"the {MLAB[(dom, m)].split(': ')[-1]} measure in the {'gratitude' if dom == 'gratitude' else 'lack-of-control'} domain, where the effect is {side} than both")
    EXC_TXT = " (the exception is " + "; ".join(parts_) + ")"
else:
    EXC_TXT = ""


def calib_table():
    out = [r"\begin{table}[htbp]", r"\centering", r"\renewcommand{\arraystretch}{1.3}",
           r"\begin{tabular}{@{} l l c c c @{}}", r"\toprule", r"\rowcolor{axisblue}",
           r"\textbf{Domain} & \textbf{Measure} & \textbf{Human} & \textbf{This study} & \textbf{Digital twin} \\",
           r"\midrule"]
    for dom, lab in (("gratitude", "Gratitude"), ("control", "Lack of Control")):
        items = [c for c in CAL if c[0] == dom]
        if dom == "control":
            out.append(r"\midrule")
        for i, (d, m, h, e, t, _) in enumerate(items):
            first = r"\multirow{3}{*}{" + lab + "}" if i == 0 else ""
            out.append(f"  {first} & {MLAB[(d, m)]} & ${h:.2f}$ & $\\mathbf{{{e:.2f}}}$ & ${t:.2f}$ \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Priming effect (prime minus control) on each measure for this study's real-prompt path and for the original study's human participants and digital twins (published values). For the lack-of-control manipulation check, the effect is control minus prime on the raw scale (higher score = greater sense of control). Original-study values are differences between the published group means.}",
            r"\label{tab:rq1-calibration}", r"\end{table}"]
    return "\n".join(out)


TAB_CALIB = calib_table()


def desc_table(w, dom, caption, label):
    mnames = ("Manip.\\ Check", "Elevation", "DV1 (Concern)") if dom == "gratitude" else ("Manip.\\ Check", "Fatigue", "DV1 (Predictability)")
    out = [r"\begin{table}[htbp]", r"\centering", r"\renewcommand{\arraystretch}{1.3}",
           r"\begin{tabular}{@{} l c c c c @{}}", r"\toprule", r"\rowcolor{axisblue}",
           r"\textbf{Condition} & \textbf{n} & \textbf{" + mnames[0] + r"} & \textbf{" + mnames[1] + r"} & \textbf{" + mnames[2] + r"} \\",
           r"\midrule"]
    order = ["real_prime", "real_control", "noinj", "dose_Low", "dose_Moderate", "dose_High", "dose_Extreme", "active_Moderate"]
    names = dict(NAME_FULL); names["real_prime"] = "Real Prime"; names["real_control"] = "Real Control"; names["noinj"] = "No-Injection Baseline"
    for c in order:
        cells = []
        for m in pr.MEASURES:
            x = w[(m, c)].dropna()
            cells.append(f"${x.mean():.2f} \\pm {x.std(ddof=1) / np.sqrt(len(x)):.2f}$")
        n = int(w[("manip_check", c)].notna().sum())
        out.append(f"{names[c]} & {n} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(out)


TAB_GRAT = desc_table(wg, "gratitude", r"Gratitude domain: mean $\pm$ SEM by condition (final successful generations; reverse-coding as described in Section~\ref{sec:measures}).", "tab:gratitude-full")
TAB_CTRL = desc_table(wc, "control", r"Lack-of-control domain: mean $\pm$ SEM by condition. Fatigue items 3, 7, 8 and Desire-for-Predictability items 1, 4, 5 are reverse-coded.", "tab:control-full")

TAB_DOSE = key_table(
    [("gratitude", "Gratitude", [("dose_Low", "real_prime", "manip_check"), ("dose_Extreme", "real_prime", "manip_check"),
                                  ("dose_Low", "real_prime", "elevation"), ("dose_Moderate", "real_prime", "elevation"),
                                  ("real_prime", "real_control", "elevation"), ("real_prime", "real_control", "dv1")]),
     ("control", "Lack of Control", [("dose_Low", "real_prime", "manip_check"), ("dose_Low", "real_prime", "elevation"),
                                      ("active_Moderate", "real_control", "manip_check"),
                                      ("real_prime", "real_control", "elevation"), ("real_prime", "real_control", "dv1")])],
    r"Key paired comparisons for RQ3 (injection vs.\ real-prompt effect size) and the real-condition effects underlying RQ4. "
    r"Differences are first condition minus second; $d_z$ is the mean within-persona difference divided by its standard deviation; "
    r"$p_{\text{Holm}}$ is adjusted over the 39 comparisons per domain listed in Appendix~\ref{app:grid}.",
    "tab:key-stats-dose", "4.1cm")

TAB_ANOM = key_table(
    [("gratitude", "Gratitude", [("noinj", "real_control", "elevation"), ("noinj", "real_control", "dv1"),
                                  ("active_Moderate", "real_prime", "dv1"), ("active_Moderate", "real_control", "dv1")]),
     ("control", "Lack of Control", [("noinj", "real_control", "elevation"), ("noinj", "real_prime", "elevation"),
                                      ("active_Moderate", "real_prime", "dv1"), ("active_Moderate", "real_control", "dv1")])],
    r"Paired comparisons of the No-Injection baseline and the active-control arm (Moderate) with the real-prompt conditions on the proximal and distal measures. "
    r"Conventions as in Table~\ref{tab:key-stats-dose}.",
    "tab:key-stats-anomaly", "4.7cm")


def appendix_grid():
    parts = [r"\section{All Paired Comparisons}", r"\label{app:grid}", "",
             r"Table~\ref{tab:grid-gratitude} and Table~\ref{tab:grid-control} list every comparison of each injection condition "
             r"(including No-Injection) with both real-prompt conditions on all three measures, plus real prime versus real control: "
             r"39 comparisons per domain. $\Delta$ is the mean within-persona difference (first minus second), $n$ the number of personas "
             r"with successful final generations in both conditions, $r$ the correlation between the two conditions' scores, "
             r"$p_{\text{Holm}}$ the Holm-adjusted paired $p$-value over the 39 comparisons of that domain, and $p_{\text{Welch}}$ "
             r"the uncorrected $p$ of Welch's independent-samples test (a sensitivity check that ignores the pairing).", ""]
    for dom, title in (("gratitude", "Gratitude domain"), ("control", "Lack-of-control domain")):
        parts += [r"{\footnotesize", r"\setlength{\tabcolsep}{3pt}",
                  r"\begin{longtable}{@{} l l r r r r r r r r @{}}",
                  r"\caption{" + title + r": all paired comparisons.}\label{tab:grid-" + dom + r"}\\",
                  r"\toprule", r"\rowcolor{axisblue}",
                  r"\textbf{Comparison} & \textbf{Measure} & $n$ & $\Delta$ & $d_z$ & $t$ & $p$ & $p_{\text{Holm}}$ & $p_{\text{Welch}}$ & $r$ \\",
                  r"\midrule", r"\endfirsthead", r"\toprule", r"\rowcolor{axisblue}",
                  r"\textbf{Comparison} & \textbf{Measure} & $n$ & $\Delta$ & $d_z$ & $t$ & $p$ & $p_{\text{Holm}}$ & $p_{\text{Welch}}$ & $r$ \\",
                  r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
        g = G[G.domain == dom]
        for _, r in g.iterrows():
            mname = {"manip_check": r"manip.\ check", "elevation": "elevation" if dom == "gratitude" else "fatigue",
                     "dv1": "dv1"}[r.measure]
            cmp_ = f"{NAME_SHORT[r.a]} vs.\\ {NAME_SHORT[r.b]}"
            parts.append(f"{cmp_} & {mname} & {int(r.n_pairs)} & ${num(r['diff'])}$ & ${num(r.dz)}$ & ${num(r.t_paired)}$ "
                         f"& {pbare(r.p_paired)} & {pbare(r.p_paired_holm)} & {pbare(r.p_welch)} & ${num(r.r)}$ \\\\")
        parts += [r"\end{longtable}", "}", ""]
    parts += [r"Table~\ref{tab:within-grid} lists the within-path comparisons: each injection condition minus No-Injection, which share the same prompt and output structure (15 comparisons per domain, Holm-adjusted within domain).", "",
              r"{\footnotesize", r"\setlength{\tabcolsep}{3pt}",
              r"\begin{longtable}{@{} l l l r r r r r r @{}}",
              r"\caption{Within-path paired comparisons: injection condition minus No-Injection.}\label{tab:within-grid}\\",
              r"\toprule", r"\rowcolor{axisblue}",
              r"\textbf{Domain} & \textbf{Condition} & \textbf{Measure} & $n$ & $\Delta$ & $d_z$ & $t$ & $p$ & $p_{\text{Holm}}$ \\",
              r"\midrule", r"\endfirsthead", r"\toprule", r"\rowcolor{axisblue}",
              r"\textbf{Domain} & \textbf{Condition} & \textbf{Measure} & $n$ & $\Delta$ & $d_z$ & $t$ & $p$ & $p_{\text{Holm}}$ \\",
              r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in WD.iterrows():
        mname = {"manip_check": r"manip.\ check", "elevation": "proximal", "dv1": "distal"}[r.measure]
        dname = "Gratitude" if r.domain == "gratitude" else "Lack of control"
        parts.append(f"{dname} & {NAME_SHORT[r.a]} & {mname} & {int(r.n_pairs)} & ${num(r['diff'])}$ & ${num(r.dz)}$ & ${num(r.t_paired)}$ & {pbare(r.p_paired)} & {pbare(r.p_paired_holm)} \\\\")
    parts += [r"\end{longtable}", "}", ""]
    return "\n".join(parts)


# ---------------- text with placeholders ----------------
RESULTS = r"""\section{Results}

\paragraph{Statistical conventions.} Every persona appears in every condition, so all comparisons are paired by persona and use the personas whose final generations succeeded in both conditions; $t$ has $n_{\text{pairs}}-1$ degrees of freedom and $d_z$ is the mean within-persona difference divided by its standard deviation. $p$-values in the text are uncorrected unless a Holm-adjusted value is given; Holm adjustment is computed over the 39 comparisons per domain listed in Appendix~\ref{app:grid}, which also reports Welch's independent-samples test as a sensitivity check. Scale scores are item means with the reverse-coding of the original analysis (Section~\ref{sec:measures}). Injection strength is referred to by the labels in Table~\ref{tab:beta-labels}.

\subsection{Comparison Axis 1: Model-Scale Calibration (RQ1)}

Table~\ref{tab:rq1-calibration} compares the priming effect on each measure in this study's real-prompt path with the effects published for the original study's human participants and digital twins.

<<TAB_CALIB>>

This study's effect falls between the human effect and the original digital-twin effect in <<C|n_between>> of the 6 comparisons<<C|exceptions>>. On the manipulation checks and on the lack-of-control proximal measure, the small open-weight model therefore over-responds to priming relative to humans but less than the original larger, closed-source model. On the distal measures all effects are small in all three sources (at most 0.25 in absolute value). The comparison uses published group means, so it is descriptive; it provides an external benchmark against which the injection results below can be read.

\subsection{Success Rate as an Indicator of Over-Injection}

Table~\ref{tab:success-rate} reports the parsing success rate for the dose injection arm in both domains, by strength.

\begin{table}[htbp]
\centering
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{@{} l c c c c @{}}
\toprule
\rowcolor{axisblue}
\textbf{Domain} & \textbf{Low} & \textbf{Moderate} & \textbf{High} & \textbf{Extreme} \\
\midrule
Gratitude & 100\% & 100\% & 97\% & \textbf{72\%} \\
Lack of Control & 97\% & 93\% & 91\% & \textbf{77\%} \\
\bottomrule
\end{tabular}
\caption{Parsing success rate for the dose injection arm, by domain and injection strength.}
\label{tab:success-rate}
\end{table}

Success rate declines monotonically with injection strength in both domains, dropping sharply at Extreme. Because this decline reflects only whether a generation could be structurally parsed, it is an outcome-independent, process-level indicator of over-injection: as strength increases, generation becomes less stable regardless of any particular downstream measure. It parallels the widening standard errors at High and Extreme (Tables~\ref{tab:gratitude-full} and~\ref{tab:control-full}), which suggests a common cause -- injection strength pushing the model outside its stable operating regime. Results at Extreme are conditional on the generations that remained parseable. By contrast, the active-control injection arm has among the \emph{highest} success rates in the dataset: 100\% at both tested strengths in the gratitude domain, and 97\% (No-Injection) and 99\% (Moderate) in the lack-of-control domain, so shifts in that arm cannot be attributed to selective attrition.

\subsection{Injection Induces Dose-Dependent Effects on the Manipulation Check (RQ2)}

Figures~\ref{fig:dose-gratitude} and~\ref{fig:dose-control} show the dose-response pattern for both domains across all three measures. On the manipulation check, both domains show a monotonic dose-response relationship in the dose injection arm. In the gratitude domain, the dose injection arm at Low already falls significantly below the real prime condition (<<S|gratitude|dose_Low|real_prime|manip_check|full>>), and at Extreme it exceeds it (<<S|gratitude|dose_Extreme|real_prime|manip_check|full>>). In the lack-of-control domain the same pattern appears in the opposite direction: the dose injection arm at Low already exceeds the real prime effect (<<S|control|dose_Low|real_prime|manip_check|full>>). Vector injection therefore induces a measurable, dose-dependent shift on the manipulation check, which answers RQ2 in the affirmative.

<<TAB_GRAT>>

<<TAB_CTRL>>

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{dose_response_gratitude.png}
\caption{Dose-response curves for the gratitude domain across the three measures. Dotted lines indicate the real-prompt reference conditions ($\pm$ SEM band), the dashed line the No-Injection baseline; the dose injection arm is a connected curve, the active-control injection arm an unconnected marker.}
\label{fig:dose-gratitude}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{dose_response_control.png}
\caption{Dose-response curves for the lack-of-control domain across the three measures, using the same conventions as Figure~\ref{fig:dose-gratitude}.}
\label{fig:dose-control}
\end{figure}

\subsection{Injection Strength vs.\ Real-Prompt Effect Size: Matching Points and Overshoot (RQ3)}

Table~\ref{tab:key-stats-dose} summarizes the key comparisons for this section and the next.

<<TAB_DOSE>>

In the gratitude domain, the dose curve crosses the real prime mean on the manipulation check at $\beta\approx<<BOOT_est>>$ (95\% bootstrap interval <<BOOT_lo>>--<<BOOT_hi>>, resampling personas; linear interpolation between neighbouring tested strengths), i.e., between Moderate and High, undershooting before and overshooting after. This location is approximate, since the strength grid was itself informed by the manipulation check (Section~\ref{sec:extraction}). In the lack-of-control domain the dose injection arm already exceeds the real-prompt effect at Low, so no matching point falls within the tested range. The overshoot is not confined to the manipulation check: on the proximal measure the dose arm exceeds the real prime condition from Moderate in the gratitude domain (<<S|gratitude|dose_Moderate|real_prime|elevation|full>>) and already at Low in the lack-of-control domain (<<S|control|dose_Low|real_prime|elevation|full>>). The two directions differ in how strongly they overshoot on the manipulation check: the lack-of-control direction produces a large overshoot (<<S|control|dose_Low|real_prime|manip_check|dz>>), whereas the in-control direction's overshoot at Moderate is much smaller (<<S|control|active_Moderate|real_control|manip_check|full>>) and does not survive Holm adjustment (<<S|control|active_Moderate|real_control|manip_check|holm>>). Injection is therefore not uniformly more potent in both directions.

\subsection{Effect Structure: Valence and Proximal--Distal Pattern (RQ4)}
\label{sec:rq4}

\paragraph{Proximal measures.} The real-prompt path reproduces the proximal spillover effect of the original study in both domains: prime exceeds control on Elevation (<<S|gratitude|real_prime|real_control|elevation|full>>) and on Fatigue (<<S|control|real_prime|real_control|elevation|full>>). At Low, the dose injection arm does not differ significantly from the real prime condition on Elevation (<<S|gratitude|dose_Low|real_prime|elevation|full>>; Table~\ref{tab:key-stats-dose}); this is the closest agreement in the study, although a non-significant difference is not evidence of equivalence and we did not run an equivalence test. In the lack-of-control domain, injection does not match the real-prompt Fatigue effect at any tested strength: it overshoots from Low (<<S|control|dose_Low|real_prime|elevation|full>>).

\paragraph{Distal measures.} In both domains the real prime and real control conditions do not differ significantly on the distal measure (gratitude: <<S|gratitude|real_prime|real_control|dv1|dzp>>; lack of control: <<S|control|real_prime|real_control|dv1|dzp>>), in line with the original study's finding that distal effects were small or non-significant for digital twins (e.g., $p=.082$ for empathic concern).

\paragraph{Differences between injection-path and real-prompt conditions.} Table~\ref{tab:key-stats-anomaly} compares the No-Injection baseline and the active-control arm with the real-prompt conditions on the proximal and distal measures. The No-Injection baseline involves no vector injection, so its differences from the real-prompt conditions cannot be due to the injected direction. It differs from the real control condition on the proximal measure in both domains and with the same sign: Elevation is higher in the gratitude domain (<<S|gratitude|noinj|real_control|elevation|full>>) and Fatigue is higher in the lack-of-control domain (<<S|control|noinj|real_control|elevation|full>>), where it is indistinguishable from the primed condition (<<S|control|noinj|real_prime|elevation|tp>>). On the distal measure it is lower than real control in the gratitude domain (<<S|gratitude|noinj|real_control|dv1|full>>) and does not differ in the lack-of-control domain (<<S|control|noinj|real_control|dv1|tp>>); the manipulation check does not differ from real control in the gratitude domain (<<S|gratitude|noinj|real_control|manip_check|tp>>). The active-control arm at Moderate is lower than both real conditions on the distal measure in both domains (gratitude: <<S|gratitude|active_Moderate|real_prime|dv1|dz>> and <<S|gratitude|active_Moderate|real_control|dv1|dz>>; lack of control: <<S|control|active_Moderate|real_prime|dv1|dz>> and <<S|control|active_Moderate|real_control|dv1|dz>>).

<<TAB_ANOM>>

\paragraph{Separating the vector from the pipeline.} These differences mix two sources: the injected vector, and the change of pipeline (the injection path omits the writing task and, with it, part of the prompt, some of the questions, and part of the output structure). The injection path, however, contains its own structure-matched baseline: No-Injection and the injection conditions share the same prompt and output structure and differ only in the injected vector. Table~\ref{tab:within-path} therefore compares the active-control arm with No-Injection.

<<TAB_WITHIN>>

The result differs by domain. In the lack-of-control domain the in-control direction moves the manipulation check in the intended direction (<<W|control|active_Moderate|manip_check|full>>) and lowers the distal measure (<<W|control|active_Moderate|dv1|full>>), while the proximal measure does not change reliably (<<W|control|active_Moderate|elevation|tp>>). Lowering the desire for predictability under an in-control direction is the direction one would expect if lack of control raises that desire, but the real-prompt contrast on this measure is null, so here injection moves a measure that the real manipulation does not. In the gratitude domain the active-control arm does not differ reliably from No-Injection on any measure, including its own target (manipulation check: <<W|gratitude|active_Moderate|manip_check|tp>>; proximal: <<W|gratitude|active_Moderate|elevation|tp>>; distal: <<W|gratitude|active_Moderate|dv1|tp>>), so the differences between that arm and the real conditions in that domain come mostly from the pipeline change. Within the injection path the dose arm raises the proximal measure strongly in both domains (gratitude, Moderate: <<W|gratitude|dose_Moderate|elevation|dz>>; lack of control, Low: <<W|control|dose_Low|elevation|dz>>) and moves the distal measure only inconsistently across strengths (Appendix Table~\ref{tab:within-grid}).

\paragraph{Persona-level agreement.} Paired scores are moderately correlated across conditions. Between real prime and real control, $r$ on the proximal measure is <<S|gratitude|real_prime|real_control|elevation|r>> (gratitude) and <<S|control|real_prime|real_control|elevation|r>> (lack of control). Between the dose arm at Low and real prime on the manipulation check, $r$ is <<S|gratitude|dose_Low|real_prime|manip_check|r>> (gratitude) and <<S|control|dose_Low|real_prime|manip_check|r>> (lack of control). We have no human or twin benchmark for these values and do not interpret them as evidence of individual-level fidelity.
"""

DISCUSSION = r"""\section{Discussion}
\label{sec:discussion}

Returning to each research question:

\paragraph{RQ1: Does the small open-weight model's real-prompt effect fall in a sensible range relative to the original study?}
Yes. The priming effect of this study's real-prompt path falls between the human effect and the original digital-twin effect in <<C|n_between>> of 6 comparisons<<C|exceptions>> (Table~\ref{tab:rq1-calibration}), and it reproduces the direction of the proximal spillover effect in both domains. The real-prompt simulation is therefore a qualitatively faithful replication of the original manipulation, and it provides an external benchmark against which the injection results can be judged.

\paragraph{RQ2: Can vector injection induce a measurable, dose-dependent affective state?}
Yes. In both domains, injection strength produced a monotonic, highly significant shift on the manipulation check (Table~\ref{tab:key-stats-dose}), so activation-space injection can induce a state, at least as measured by direct self-report items.

\paragraph{RQ3: How does injection strength compare to real-prompt effect size?}
The two can be matched on the manipulation check at a specific strength -- in the gratitude domain, at approximately $\beta\approx<<BOOT_est>>$ (95\% bootstrap interval <<BOOT_lo>>--<<BOOT_hi>>) -- but beyond it injection overshoots, on the proximal measure as well as on the manipulation check, and the two directions are not symmetric: the in-control direction's effect on its own target is much smaller (<<S|control|active_Moderate|real_control|manip_check|dz>>) and does not survive Holm adjustment. The calibration is direction- and measure-specific rather than a single global property of the method.

\paragraph{RQ4: Does injection reproduce the same valence-dependent, proximal--distal effect structure as real-prompt priming?}
Partially. The real-prompt path shows a proximal effect and a null distal effect in both domains, as in the original study. Injection reproduces the proximal effect only at one strength in one domain (Low, gratitude Elevation, where it is not significantly different from the real prompt) and overshoots it elsewhere, and it moves the distal measure in conditions where the real manipulation does not.

\subsection{What Injection Appears to Be Doing}

Taken together, the RQ1 calibration and the RQ3 overshoot suggest a specific interpretation. Digital twins are known from the original study to over-respond to prompt-based priming relative to humans, and our real-prompt path reproduced that tendency at a magnitude between the human and the original large-model twin (Table~\ref{tab:rq1-calibration}). Vector injection does not reproduce this tendency through a different channel; it exceeds it, on the manipulation check and on the proximal measure. This is more consistent with injection \emph{amplifying} an existing property of the digital-twin paradigm -- heightened sensitivity to affective manipulation relative to a human baseline -- than with injection reconstructing the psychological state that real writing induces. Agreement with the real-prompt path is at the level of means; persona-level correlations are only moderate (Section~\ref{sec:rq4}).

The asymmetry between directions complicates the picture. In the lack-of-control domain the in-control direction has a comparatively small effect on the manipulation check against the real control condition (<<S|control|active_Moderate|real_control|manip_check|dz>>), yet within the injection path it lowers the distal measure (<<W|control|active_Moderate|dv1|dz>> relative to No-Injection), a measure on which the real manipulation has no reliable effect. A direction that moves a downstream dimension more than the real manipulation does is difficult to reconcile with the assumption that it reproduces the real manipulation. One possibility is that the contrastive writing corpus mixes topical and affective signal, and that the mixture is not symmetric across the two poles: writing about a lack of control or about gratitude may carry a more concentrated, extractable affective signal than writing about a routine or an in-control episode. The extraction also averaged over the writing instructions themselves (Section~\ref{sec:extraction}), which differ in wording between the poles, and in the gratitude domain the linear probe separated the two classes perfectly at every candidate layer, as expected when topic and instruction wording are available (\todo{confirm for the lack-of-control domain}). These observations are compatible with a direction that tracks topic and wording at least as much as affect.

\subsection{Two Sources of Differences: the Vector and the Pipeline}

Differences between injection-path conditions and the real-prompt conditions have two possible sources: the injected vector, and the change of pipeline that injection entails. Because the injection path contains a structure-matched baseline (No-Injection), the two can be separated. The No-Injection baseline differs from the real control condition on the proximal measure in both domains, with the same sign (higher), so omitting the writing task raises the proximal measure relative to a real control writing task regardless of the injected vector (Table~\ref{tab:key-stats-anomaly}). The active-control arm, in contrast, adds nothing reliable beyond No-Injection in the gratitude domain, while in the lack-of-control domain it lowers the distal measure within the injection path (Table~\ref{tab:within-path}). Both sources are therefore present, and which one dominates depends on the domain and the measure.

The pipeline effect cannot be traced to a single cause with the present design. The injection path differs from the real-prompt path in the omitted writing task, a shorter prompt, fewer questions, and a different output structure. One further reason is specific to the lack-of-control proximal measure: the Fatigue items ask about ``recalling and describing the event'' earlier in the survey, and the injection path contains no such event, so that measure is only partly defined there. A control in which the model completes an affectively neutral writing task, with prompt and output structure matched to the real-prompt path, before receiving (or not receiving) injected activations would isolate the contribution of the writing task itself; we leave it for future work.

\subsection{Limitations}

This study has several limitations. First, all experiments use a single base model (Llama-3.1-8B-Instruct) and a single extraction method (MeanDiff with a three-layer window). The layer window was not chosen by a discriminating criterion, since, in the gratitude domain, the probe separated the classes perfectly at every candidate layer; whether the reported patterns generalize to other models, extraction methods, or windows is untested. Second, the active-control arm was tested at one non-zero strength per domain, so no dose-response curve is available for that direction. Third, the strength grid was informed by a three-persona pilot on the manipulation check, and those personas are part of the main sample, so the matching strength in RQ3 is approximate. Fourth, we use paired tests for each comparison but not a single mixed-effects model across all conditions, and we do not test equivalence formally, so statements that two conditions ``do not differ'' are statements about non-significance. Multiple-comparison adjustment was applied only to the comparisons in the appendix. Fifth, the real-prompt and injection paths differ in more than the injected vector, and the lack-of-control Fatigue items refer to a writing task that the injection path omits; we did not run a structure-matched control with a neutral writing task. Sixth, the retry limit differed between domains (two attempts for gratitude, three for lack of control), and results at Extreme strength are conditional on parseable generations. Finally, the vector was extracted from writing by the same pool of participants whose twins we simulated, which may give the injected direction some information about those individuals; excluding the simulated personas from the extraction corpus would remove this possibility.

\subsection{Practical Implications}

If the motivation for vector injection is to avoid generating a full written response, the present results suggest that the substitution is measure-dependent. On the manipulation check, a well-chosen strength matches the real-prompt effect in the gratitude domain, and on the gratitude proximal measure a low strength is close to it at the level of means. However, injection overshoots the real-prompt effect elsewhere, the pipeline change itself shifts the proximal measure, and in one domain the injected direction moves a distal measure on its own. Researchers considering injection as a substitute for prompt-based state induction should calibrate and validate strength separately for each outcome measure, and should include a no-injection control that matches the prompt and output format of the injection path.

\section{Conclusion}

We asked whether activation steering can substitute for the writing task in prompt-based affective priming of LLM digital twins. On the manipulation check, injection produced a dose-dependent shift that could be matched to the real-prompt effect in one domain and overshot it in the other; on the gratitude proximal measure it agreed with the real-prompt effect at low strength. Beyond that the substitution did not hold uniformly: injection amplified rather than reproduced the real effect on the manipulation check and the proximal measure, the two injection directions behaved asymmetrically, and differences from the real-prompt path arose from the vector in one domain and from the change of pipeline in both. For researchers using digital twins, the practical lesson is that validity established for one way of inducing a state does not carry over automatically to another, and that any such substitution needs structure-matched controls and measure-by-measure validation. Immediate next steps are a neutral-writing control that matches prompt and output structure, other models and extraction methods, and an extraction corpus that excludes the simulated personas.
"""


ABSTRACT = r"""\begin{abstract}
Prompt-based affective priming -- having a model write about a gratitude- or lack-of-control-related experience before completing downstream measures -- has been shown to induce measurable psychological states in LLM-based digital twins, replicating classic human affective-priming effects. This study tests whether activation steering (vector injection) can substitute for the writing task itself, skipping it entirely while injecting a pre-extracted affective direction vector at varying strengths. Using Llama-3.1-8B-Instruct, we implement two parallel pipelines -- a real-prompt simulation path that replicates the original methodology, and a vector-injection path -- across the same two affective domains (gratitude, lack of control) and the same manipulation-check, proximal, and distal measures used in the original study. The real-prompt path's priming effects fall between the original study's human and digital-twin results on most measures, providing an external calibration point. Vector injection reliably induces dose-dependent effects on the manipulation check and can be calibrated to match the real-prompt effect at a specific strength; beyond that point it overshoots, on the proximal measure as well, and it does so asymmetrically across the two injection directions. Injection agrees with real-prompt priming at the level of means only in limited cases (for example, low-strength injection on the gratitude proximal measure), and distal measures show weak effects under the real-prompt path, consistent with the original study. Comparing each injection condition with a no-injection baseline that shares the same prompt structure separates two sources of difference: omitting the writing task raises the proximal measure relative to the real-prompt control in both domains, whereas the injected direction itself moves a distal measure in the lack-of-control domain. These results indicate that vector injection can substitute for prompt-based priming on some measures, but the substitution is measure-dependent and does not transfer uniformly across the proximal--distal spectrum.
\end{abstract}
"""


def fill(text):
    def rep(m):
        key = m.group(1)
        if key == "TAB_DOSE":
            return TAB_DOSE
        if key == "TAB_ANOM":
            return TAB_ANOM
        if key == "TAB_WITHIN":
            return TAB_WITHIN
        if key == "TAB_CALIB":
            return TAB_CALIB
        if key == "TAB_GRAT":
            return TAB_GRAT
        if key == "TAB_CTRL":
            return TAB_CTRL
        if key == "C|n_between":
            return str(N_BETWEEN)
        if key == "C|exceptions":
            return EXC_TXT
        if key.startswith("W|"):
            return wstat(*key.split("|")[1:])
        if key.startswith("BOOT_"):
            return BOOT[key[5:]]
        parts = key.split("|")
        assert parts[0] == "S", key
        return stat(*parts[1:])
    out = re.sub(r"<<(.+?)>>", rep, text)
    assert "<<" not in out
    return out


for name, text in (("results_revised.tex", RESULTS), ("discussion_revised.tex", DISCUSSION),
                   ("abstract_revised.tex", ABSTRACT)):
    open(name, "w").write(fill(text))
open("appendix_grid.tex", "w").write(appendix_grid())
print("written: results_revised.tex discussion_revised.tex abstract_revised.tex appendix_grid.tex")
