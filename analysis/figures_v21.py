"""figures_v21.py: saturation statistics and Figures 2-4 for 'Differences between single-drift and
double-drift illusions' (Saleki, Cavanagh & Tse), revision after Patrick's v20 (15 Oct 2026).

Inputs : the v16 package data (data/preproc/speed-2.5_df-ori.csv, speed-2.5_df-len.csv) with the v16
         rules: every orientation setting is a report, 3-SD screen within participant x duration
         (orientation) and within participant x stimulus x duration (length); Eq. 1 for the
         double-drift offset. Literature values: figure4_points_v21.csv (dva).
Outputs: Figure2.png/pdf, Figure3.png/pdf, Figure4.png/pdf, results_v21.json, cell_means_v21.csv
Usage  : python figures_v21.py <preproc_dir> <literature_csv> <out_dir>
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRE, LIT, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
DURS = [50, 200, 350, 500, 650, 800]
ORANGE, PURPLE, GREY = "#E07B00", "#7B3FB5", "#3F3F3F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.linewidth": 1.2,
                     "pdf.fonttype": 42})
R = {}

def z_within(df, val, by):
    g = df.groupby(by)[val]
    return (df[val] - g.transform("mean")) / g.transform("std")

# ---------------- data, v16 rules ----------------
ori = pd.read_csv(os.path.join(PRE, "speed-2.5_df-ori.csv"))
ln = pd.read_csv(os.path.join(PRE, "speed-2.5_df-len.csv"))
ori["Z"] = z_within(ori, "ANGLE_FROM_VERTICAL", ["SUB_ID", "DURATION_MS"])
ori = ori[~(ori["Z"].abs() > 3)]
ang = ori.groupby(["SUB_ID", "DURATION_MS"])["ANGLE_FROM_VERTICAL"].mean().unstack()[DURS]
R["angle_mean"] = {d: float(ang[d].mean()) for d in DURS}; R["angle_sem"] = {d: float(ang[d].sem()) for d in DURS}

ln["Z"] = z_within(ln, "REPORT_LENGTH_DVA", ["SUB_ID", "STIM_TYPE", "DURATION_MS"])
lc = ln[~(ln["Z"].abs() > 3)].copy()
lc["OFFSET"] = np.where(lc["STIM_TYPE"] == "dd", lc["REPORT_LENGTH_DVA"] * np.sin(np.deg2rad(lc["ANGLE_FROM_VERTICAL"])), lc["REPORT_LENGTH_DVA"])
cell = lc.groupby(["SUB_ID", "STIM_TYPE", "DURATION_MS"]).agg(OFFSET=("OFFSET", "mean"), N=("OFFSET", "size")).reset_index()
cell.to_csv(os.path.join(OUT, "cell_means_v21.csv"), index=False)
piv = {s: cell[cell.STIM_TYPE == s].pivot(index="SUB_ID", columns="DURATION_MS", values="OFFSET")[DURS] for s in ["deva", "dd"]}
piv["both"] = (piv["deva"] + piv["dd"]) / 2
NAMES = {"deva": "single-drift", "dd": "double-drift", "both": "average of the two"}

# ---------------- saturation ----------------
def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj
LIN = np.array([-5, -3, -1, 1, 3, 5]); QUAD = np.array([5, -1, -4, -4, -1, 5])   # orthogonal polynomials, 6 equally spaced levels
expo = lambda t, A, tau: A * (1 - np.exp(-t / tau))
sat = {}
for s, X in piv.items():
    steps, ps = [], []
    for a, b in zip(DURS[:-1], DURS[1:]):
        d = X[b] - X[a]; t, p = stats.ttest_rel(X[b], X[a]); lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        steps.append(dict(step=f"{a}-{b}", increase=float(d.mean()), ci95=[float(lo), float(hi)], t=float(t), df=len(d) - 1, p=float(p), n_increasing=int((d > 0).sum())))
        ps.append(p)
    for st, pa in zip(steps, holm(ps)): st["p_holm"] = float(pa)
    con = {}
    for nm, c in [("linear", LIN), ("quadratic", QUAD)]:
        v = X.values @ c; t, p = stats.ttest_1samp(v, 0)
        con[nm] = dict(mean=float(v.mean()), t=float(t), df=len(v) - 1, p=float(p), n_negative=int((v < 0).sum()))
    taus = []
    for _, row in X.iterrows():
        (A_, tau), _ = curve_fit(expo, np.array(DURS, float), row.values, p0=[1.5, 400], bounds=([0, 10], [20, 1e5])); taus.append(float(tau))
    (Ag, tg), _ = curve_fit(expo, np.array(DURS, float), X.mean().values, p0=[1.5, 400])
    sat[s] = dict(steps=steps, contrasts=con, tau_by_participant=taus, tau_median=float(np.median(taus)),
                  tau_range=[float(min(taus)), float(max(taus))], group_fit=dict(A=float(Ag), tau=float(tg)),
                  fraction_of_asymptote_at_800=float(1 - np.exp(-800 / tg)), means=[float(m) for m in X.mean().values],
                  sems=[float(m) for m in X.sem().values])
# does the curvature differ between the two stimuli?
qd = piv["deva"].values @ QUAD - piv["dd"].values @ QUAD; t, p = stats.ttest_1samp(qd, 0)
R["quadratic_sd_minus_dd"] = dict(mean=float(qd.mean()), t=float(t), p=float(p))
R["saturation"] = sat

# ---------------- Figure 2 ----------------
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
for a_, lab in zip(axes, ["(a)", "(b)"]): a_.text(-0.18, 1.04, lab, transform=a_.transAxes, fontsize=22)
ax = axes[0]
ax.errorbar(DURS, [ang[d].mean() for d in DURS], yerr=[ang[d].sem() for d in DURS], color=PURPLE, lw=3, marker="o", ms=9, capsize=0, label="Double-drift")
ax.set_xticks(DURS); ax.set_ylim(40, 90); ax.set_yticks([40, 50, 60, 70, 80, 90]); ax.set_xlabel("Path duration (ms)"); ax.set_ylabel("Illusory angle (°)")
ax.legend(frameon=False, loc="lower left", fontsize=13)
ax = axes[1]
for s, col, mk, lab in [("deva", ORANGE, "o", "Single-drift"), ("dd", PURPLE, "D", "Double-drift")]:
    X = piv[s]; off = -6 if s == "deva" else 6
    ax.errorbar(np.array(DURS) + off, X.mean().values, yerr=X.sem().values, color=col, lw=3, marker=mk, ms=8 if mk == "o" else 7, capsize=0, label=lab, zorder=4)
ax.set_xticks(DURS); ax.set_ylim(0, 1.5); ax.set_yticks(np.arange(0, 1.51, 0.25)); ax.set_xlabel("Path duration (ms)"); ax.set_ylabel("Illusory offset (dva)")
ax.legend(frameon=False, loc="upper left", fontsize=13, handlelength=2.6)
fig.tight_layout(w_pad=3)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"Figure2.{ext}"), dpi=300)
plt.close(fig)

# ---------------- Figure 3 ----------------
sm = pd.DataFrame({s: piv[s].mean(axis=1) for s in ["deva", "dd"]})
r, p = stats.pearsonr(sm["deva"], sm["dd"]); R["corr"] = dict(r=float(r), p=float(p))
fig, ax = plt.subplots(figsize=(6.0, 5.8)); lim = (0.4, 1.2)
ax.plot(lim, lim, ls=":", color="#777777", lw=1.5, zorder=1)
ax.scatter(sm["deva"], sm["dd"], s=55, color="black", zorder=3)
ax.text(0.05, 0.95, f"r = {r:.2f}, p = {p:.3f}", transform=ax.transAxes, va="top", fontsize=14)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.set_xticks(np.arange(0.4, 1.21, 0.2)); ax.set_yticks(np.arange(0.4, 1.21, 0.2))
ax.set_xlabel("Single-drift offset (dva)"); ax.set_ylabel("Double-drift offset (dva)")
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"Figure3.{ext}"), dpi=300)
plt.close(fig)

# ---------------- Figure 4 ----------------
lit = pd.read_csv(LIT)
sd800, dd800 = piv["deva"][800], piv["dd"][800]
ours = {"ours_sd": (sd800.mean(), sd800.min(), sd800.max()), "ours_dd": (dd800.mean(), dd800.min(), dd800.max())}
R["fig4_ours_800"] = {k: dict(mean=float(v[0]), lo=float(v[1]), hi=float(v[2])) for k, v in ours.items()}
with plt.rc_context({"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12}):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    dv = lit[lit.study.str.startswith("De Valois")]
    ax.plot(dv.ecc_dva, dv.shift_dva, "-", color=GREY, lw=1.4, zorder=2)
    def rng(x, lo, hi, col, w=0.16):
        ax.plot([x, x], [lo, hi], color=col, lw=1.6, zorder=3); ax.plot([x - w, x + w], [lo, lo], color=col, lw=1.3, zorder=3); ax.plot([x - w, x + w], [hi, hi], color=col, lw=1.3, zorder=3)
    for _, row in lit.iterrows():
        x, y = row.ecc_dva, row.shift_dva
        if row.kind == "one":
            if not np.isnan(row.range_lo): rng(x, row.range_lo, row.range_hi, GREY)
            ax.scatter(x, y, s=60, color=GREY, zorder=4)
        elif row.kind == "rel":
            if not np.isnan(row.range_lo): rng(x, row.range_lo, row.range_hi, GREY)
            ax.scatter(x, y, s=60, facecolor="white", edgecolor=GREY, lw=1.5, zorder=4)
        elif row.kind == "dd_lit": ax.scatter(x, y, s=190, marker="*", color=PURPLE, zorder=5)
    for k, x, col, face in [("ours_sd", 10.8, ORANGE, ORANGE), ("ours_dd", 11.4, PURPLE, "white")]:
        m, lo, hi = ours[k]; rng(x, lo, hi, col); ax.scatter(x, m, s=105, marker="D", facecolor=face, edgecolor=col, lw=2, zorder=6)
    for _, row in lit.iterrows():
        if isinstance(row.label, str) and row.label:
            ax.annotate(row.label, (row.ecc_dva, row.shift_dva), xytext=(row.lab_dx, row.lab_dy), textcoords="offset points",
                        fontsize=10, color=GREY, ha=row.lab_ha, va="center")
    ax.set_yscale("log"); ax.set_ylim(0.02, 7); ax.set_yticks([0.03, 0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
    ax.set_xlim(-0.6, 14.6); ax.set_xticks(range(0, 15, 2))
    ax.set_xlabel("Eccentricity (dva)"); ax.set_ylabel("Position shift of one patch (dva)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=GREY, ms=8, label="Single patch, one-sided"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, mew=1.5, ms=8, label="Two patches, value halved"),
         Line2D([], [], marker="D", ls="", color=ORANGE, ms=8, label="This study, single-drift (800 ms)"),
         Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=PURPLE, mew=2, ms=8, label="This study, double-drift (800 ms)"),
         Line2D([], [], marker="*", ls="", color=PURPLE, ms=13, label="Lisi & Cavanagh (2015), Exp. 2, double-drift")]
    ax.legend(handles=h, frameon=False, loc="upper left", fontsize=10.5, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"Figure4.{ext}"), dpi=300)
    plt.close(fig)

json.dump(R, open(os.path.join(OUT, "results_v21.json"), "w"), indent=1, default=float)
print(json.dumps({s: dict(last_step=sat[s]["steps"][-1], quad=sat[s]["contrasts"]["quadratic"], tau=sat[s]["tau_median"]) for s in sat}, indent=1))
print("corr", R["corr"], "quad diff", R["quadratic_sd_minus_dd"])
