"""figure1_v11.py: Figure 1 of 'Accumulation of position errors in internally drifting stimuli' (v16).
Two panels, start and end patches only, dots for the intermediate positions, no position labels. Physical patches are drawn at full
contrast, perceived (illusory) end patches dimmer. In (b) the physical endpoint sits where the perceived path's projection on the
physical path ends (D = L cos theta), so the red shift arrow is perpendicular to the physical path. Usage: python figure1_v11.py <out_dir>"""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Wedge, Arc, Polygon
from matplotlib.lines import Line2D

YEL, GREY, PURPLE, BLACK, RED = "#fcee21", "#999999", "#be39d6", "#333333", "#ed1c24"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

def gabor(ax, x, y, R=0.46, theta_deg=0.0, cycles=1.6, contrast=0.8, zorder=5, n=300, dim=False):
    """Striped disc with a soft edge; grating orientation theta_deg (0 = horizontal stripes)."""
    g = np.linspace(-1, 1, n); X, Y = np.meshgrid(g, g); r = np.sqrt(X**2 + Y**2)
    th = np.deg2rad(theta_deg); u = -X * np.sin(th) + Y * np.cos(th)            # coordinate across the stripes
    if dim: contrast, base, amax = contrast * 0.45, 0.78, 0.75                     # perceived (illusory) position: lighter, lower contrast
    else: base, amax = 0.62, 1.0
    lum = base + 0.3 * contrast * np.cos(2 * np.pi * cycles * u)
    alpha = amax * np.clip((1 - r) / 0.18, 0, 1) ** 1.2                           # opaque disc, soft rim
    rgba = np.dstack([lum, lum, lum, alpha])
    ax.imshow(rgba, extent=[x - R, x + R, y - R, y + R], zorder=zorder, interpolation="bilinear", origin="lower")

def arrow(ax, p0, p1, color, lw=3.2, style="-|>", ms=22, zorder=6, shrink=0.0):
    (x0, y0), (x1, y1) = p0, p1
    d = np.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / d, (y1 - y0) / d
    a_ = FancyArrowPatch((x0 + ux * shrink, y0 + uy * shrink), (x1 - ux * shrink, y1 - uy * shrink), arrowstyle=style, mutation_scale=ms,
                         color=color, lw=lw, zorder=zorder, shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt")
    ax.add_patch(a_)

def dots(ax, p0, p1, fracs=(0.36, 0.5, 0.64), color="#555555", size=9):
    (x0, y0), (x1, y1) = p0, p1
    ax.scatter([x0 + f * (x1 - x0) for f in fracs], [y0 + f * (y1 - y0) for f in fracs], s=size**2, color=color, zorder=6, linewidths=0)

def internal_arrow(ax, x, y, direction_deg, color=YEL, L=0.46):
    th = np.deg2rad(direction_deg); dx, dy = L * np.cos(th), L * np.sin(th)
    arrow(ax, (x - dx / 2, y - dy / 2), (x + dx / 2, y + dy / 2), color, lw=3.4, ms=20, zorder=8)

def panel_label(ax, s):
    ax.text(-0.02, 1.0, s, transform=ax.transAxes, fontsize=24, va="top", ha="left")

def main(out):
    fig = plt.figure(figsize=(12.0, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 7.2], hspace=0.05, wspace=0.08)
    # ---------------- legend
    lax = fig.add_subplot(gs[0, :]); lax.set_xlim(0, 12); lax.set_ylim(0, 2); lax.axis("off")
    lax.add_patch(plt.Rectangle((0.05, 0.08), 11.9, 1.84, fill=False, ec="#cccccc", lw=1.4, zorder=1, joinstyle="round"))
    def entry(x, y, kind, text):
        if kind == "arrow_y": arrow(lax, (x, y), (x + 0.9, y), YEL, lw=3.4, ms=20)
        if kind == "arrow_g": arrow(lax, (x, y), (x + 0.9, y), GREY, lw=3.2, ms=20)
        if kind == "arrow_k": arrow(lax, (x, y), (x + 0.9, y), BLACK, lw=3.2, ms=20)
        if kind == "arrow_r": arrow(lax, (x, y), (x + 0.9, y), RED, lw=3.2, style="<|-|>", ms=20)
        if kind == "arc":
            lax.add_patch(Arc((x + 0.45, y - 0.32), 1.0, 1.0, angle=0, theta1=30, theta2=150, color=PURPLE, lw=3.0))
            lax.add_patch(Wedge((x + 0.45, y - 0.32), 0.5, 30, 150, facecolor=PURPLE, alpha=0.18, lw=0))
        return lax.text(x + 1.1, y, text, fontsize=14, va="center", ha="left", color="#222222")
    labels = [entry(0.35, 1.45, "arrow_y", "Internal motion"), entry(3.9, 1.45, "arrow_g", "Perceived motion"), entry(7.35, 1.45, "arrow_r", "Perceived position shift"),
              entry(0.35, 0.55, "arrow_k", "External motion"), entry(3.9, 0.55, "arc", "Illusory angle")]
    fig.canvas.draw()
    for t in labels:                                                                 # every label must end inside the legend box
        x1 = lax.transData.inverted().transform(t.get_window_extent())[1, 0]
        assert x1 < 11.85, f"legend label overflows the box: {t.get_text()} ends at {x1:.2f}"
    # ---------------- (a) single-drift
    ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(-2.3, 2.7); ax.set_ylim(-1.3, 4.15); ax.set_aspect("equal"); ax.axis("off")
    S0, St = (0.0, 0.0), (0.0, 3.4)
    arrow(ax, (-0.75, 0.0), (-0.75, 3.4), GREY, lw=3.2, ms=22)                   # perceived motion
    dots(ax, S0, St, color="#b5b5b5")                                            # perceived intermediate positions
    arrow(ax, (0.85, 0.0), (0.85, 3.4), RED, lw=3.2, style="<|-|>", ms=22)       # position shift
    gabor(ax, *S0, theta_deg=0); internal_arrow(ax, *S0, 90)                     # physical position (bright)
    gabor(ax, *St, theta_deg=0, dim=True); internal_arrow(ax, *St, 90)           # perceived end position (dim)
    # axes icon
    arrow(ax, (-2.0, -1.1), (-1.3, -1.1), BLACK, lw=1.6, ms=12); arrow(ax, (-2.0, -1.1), (-2.0, -0.4), BLACK, lw=1.6, ms=12)
    ax.text(-1.22, -1.1, "x", fontsize=13, va="center"); ax.text(-2.0, -0.32, "y", fontsize=13, ha="center", va="bottom")
    panel_label(ax, "(a)")
    # ---------------- (b) double-drift
    ax = fig.add_subplot(gs[1, 1]); ax.set_xlim(-1.4, 3.6); ax.set_ylim(-1.3, 4.15); ax.set_aspect("equal"); ax.axis("off")
    ang = np.deg2rad(38.0); L = 3.4                                              # path 38 deg from horizontal: illusory angle 52 deg
    D = L * np.sin(ang)                                                          # = L cos(60 deg): projection of the perceived path on the physical path
    S0 = (0.0, 0.0); Sper = (0.0, L); Sphys = (D * np.cos(ang), D * np.sin(ang))
    # illusory angle: wedge between the physical path (30 deg) and the perceived path (90 deg)
    ax.add_patch(Wedge(S0, 0.85, 38, 90, facecolor=PURPLE, alpha=0.16, lw=0, zorder=2))
    ax.add_patch(Arc(S0, 1.7, 1.7, angle=0, theta1=38, theta2=90, color=PURPLE, lw=3.0, zorder=3))
    # perceived motion (grey, left of the vertical dots) and external motion (black, below the oblique dots)
    arrow(ax, (-0.75, 0.0), (-0.75, L), GREY, lw=3.2, ms=22)
    nx, ny = np.sin(ang), -np.cos(ang)                                           # unit normal pointing below the oblique path
    dx, dy = np.cos(ang), np.sin(ang)
    arrow(ax, (S0[0] + 0.66 * nx + 0.2 * dx, S0[1] + 0.66 * ny + 0.2 * dy), (Sphys[0] + 0.66 * nx, Sphys[1] + 0.66 * ny), BLACK, lw=3.2, ms=22)
    dots(ax, S0, Sper, color="#b5b5b5"); dots(ax, S0, Sphys, fracs=(0.48, 0.60, 0.72))
    # position shift from the physical to the perceived endpoint, perpendicular to the physical path
    assert abs((Sper[0] - Sphys[0]) * np.cos(ang) + (Sper[1] - Sphys[1]) * np.sin(ang)) < 1e-9
    arrow(ax, Sphys, Sper, RED, lw=3.2, style="<|-|>", ms=22, shrink=0.56)
    for (x, y) in (S0, Sphys): gabor(ax, x, y, theta_deg=38); internal_arrow(ax, x, y, 128)        # physical positions (bright): bars along the path, drift orthogonal
    gabor(ax, *Sper, theta_deg=38, dim=True); internal_arrow(ax, *Sper, 128)                      # perceived end position (dim)
    panel_label(ax, "(b)")
    fig.savefig(os.path.join(out, "fig1_v11.png"), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(os.path.join(out, "fig1_v11.pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1])
