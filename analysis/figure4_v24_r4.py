"""figure4_v24_r4.py (rev 4, 29 Oct 2026: smaller labels, no horizontal guides; rev 3 below): Figure 4 of 'Differences between single-drift and double-drift illusions'
(Saleki, Cavanagh & Tse) in the format Patrick proposed in v23 (25 Oct 2026), with Fu et al. (2004) added.

Format (v23): log shift axis, no legend, one colour per study, the label beside the study's points;
diamonds = one stationary patch (one-sided shift), squares = two patches drifting in opposite
directions (relative shift halved), large circles = this study at 800 ms with participant range,
stars = double-drift: Lisi & Cavanagh (2015) Exp. 2 (small) and this study (large, with participant range).
Rev 3 (Sharif): this study's two points named in the plot with arrows ('This study single-drift',
'This study double-drift'), Lisi label adds 'double-drift'; every label one font, size and colour. Literature values: figure4_points_v24.csv (dva).
This study: cell_means_v21.csv (v16 rules), 800-ms participant means.
Usage : python figure4_v24.py <cell_means_csv> <literature_csv> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELL, LIT, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
ORANGE, PURPLE, LABEL = "#D3812E", "#7443AF", "#333333"
TXT = 10.5  # one size for every in-plot label
COLOUR = {"De Valois": "#DA3B26", "Kosovicheva": "#81D653", "Chung": "#306F1D", "Ye": "#479FF8",
          "Hu": "#F2C94C", "Bressler": "#8A2052", "Whitney": "#000000", "Fu": "#D04FA6", "Lisi": "#7443AF"}
# label text, anchor point (ecc, shift), offset in points, alignment
LABELS = [("De Valois & De Valois (1991)", (0.0, 0.084), (-6, -17), "left"),
          ("Chung et al. (2007)", (1.0, 0.0333), (11, 3), "left"),
          ("Hu et al. (2022)", (4.3, 0.115), (11, 0), "left"),
          ("Ye et al. (2018)", (9.0, 0.2), (10, -3), "left"),
          ("Kosovicheva et al. (2014)", (10.0, 0.55), (2, -17), "left"),
          ("Bressler & Whitney (2006)", (9.1, 0.55), (-11, 4), "right"),
          ("Whitney et al. (2003)", (13.0, 1.8375), (-11, 4), "right"),
          ("Lisi & Cavanagh (2015)\ndouble-drift", (10.0, 1.315), (-12, -6), "right"),
          ("Fu et al. (2004)", (8.0, 0.0534), (11, -1), "left")]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.2, "pdf.fonttype": 42, "font.size": 13, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})

cell = pd.read_csv(CELL)
c800 = cell[cell.DURATION_MS == 800].pivot(index="SUB_ID", columns="STIM_TYPE", values="OFFSET")
ours = {s: (c800[s].mean(), c800[s].min(), c800[s].max()) for s in ["deva", "dd"]}
lit = pd.read_csv(LIT)

from matplotlib.colors import to_rgb
from matplotlib.ticker import NullLocator
def edge(c, k=0.42):
    """Darker shade of c for outlines (black symbols get a white rim)."""
    r, g, b = to_rgb(c)
    return "#FFFFFF" if max(r, g, b) < 0.1 else (r * (1 - k), g * (1 - k), b * (1 - k))

fig, ax = plt.subplots(figsize=(10.4, 6.4))
for _, r in lit.iterrows():
    key = next(k for k in COLOUR if r.study.startswith(k))
    x = r.ecc_dva + (0.3 if key == "Hu" else 0)   # Hu (2022) shares 4 dva with Chung (2007): drawn 0.3 dva to the right
    c = COLOUR[key]
    if r.kind == "one":
        ax.scatter(x, r.shift_dva, s=165, marker="D", color=c, edgecolor=edge(c), lw=1.2, zorder=4)
    elif r.kind == "rel":
        ax.scatter(x, r.shift_dva, s=150, marker="s", color=c, edgecolor=edge(c), lw=1.2, zorder=4)
    elif r.kind == "dd_lit":
        ax.scatter(x, r.shift_dva, s=300, marker="*", color=PURPLE, edgecolor=edge(PURPLE), lw=1.1, zorder=5)
m, lo, hi = ours["deva"]   # single-drift: circle
ax.errorbar(10.8, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=edge(ORANGE, 0.25), elinewidth=1.8, capsize=6, capthick=1.8, zorder=5)
ax.scatter(10.8, m, s=330, marker="o", color=ORANGE, edgecolor=edge(ORANGE), lw=1.6, zorder=6)
m, lo, hi = ours["dd"]     # double-drift: star, like Lisi & Cavanagh's double-drift point
ax.errorbar(11.4, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=edge(PURPLE, 0.25), elinewidth=1.8, capsize=6, capthick=1.8, zorder=5)
ax.scatter(11.4, m, s=720, marker="*", color=PURPLE, edgecolor=edge(PURPLE), lw=1.3, zorder=6)
ARROW = dict(arrowstyle="-|>", color=LABEL, lw=1.4, mutation_scale=15, shrinkA=3)
ax.annotate("This study double-drift", (11.4, ours["dd"][0]), xytext=(12.75, ours["dd"][0]),
            fontsize=TXT, color=LABEL, ha="left", va="center", annotation_clip=False,
            arrowprops=dict(ARROW, shrinkB=17))
ax.annotate("This study single-drift", (10.8, ours["deva"][0]), xytext=(12.75, 0.70),
            fontsize=TXT, color=LABEL, ha="left", va="center", annotation_clip=False,
            arrowprops=dict(ARROW, shrinkB=12, connectionstyle="arc3,rad=-0.35"))
for text, xy, off, ha in LABELS:
    ax.annotate(text, xy, xytext=off, textcoords="offset points", fontsize=TXT, color=LABEL, ha=ha, va="center", multialignment=ha)
ax.set_yscale("log"); ax.set_ylim(0.008, 6)
ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3"])
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlim(-0.6, 14.6); ax.set_xticks(range(0, 15, 2))
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#333333"); ax.spines[sp].set_linewidth(1.0)
ax.spines["left"].set_bounds(0.01, 3); ax.spines["bottom"].set_bounds(0, 14)
ax.tick_params(colors="#333333", width=1.0, length=5, direction="out")
ax.set_xlabel("Eccentricity (dva)", labelpad=8, color="#333333"); ax.set_ylabel("Position shift of one patch (dva)", labelpad=8, color="#333333")
fig.subplots_adjust(left=0.09, right=0.82, bottom=0.12, top=0.97)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Figure4_r4.{ext}"), dpi=300)
print({k: [round(v, 3) for v in t] for k, t in ours.items()})
