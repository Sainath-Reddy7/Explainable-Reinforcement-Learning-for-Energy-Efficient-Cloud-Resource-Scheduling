"""
plot_architecture.py — Publication-format architecture figure.

Format targets (Nature / Nature Machine Intelligence style):
  * double-column width, 183 mm (7.2 in), vector PDF + SVG + 300 dpi PNG
  * NPG colour palette, muted tints, hairline strokes (0.6-1.0 pt)
  * print-calibrated type: panel letters 9 pt bold, labels 6-7.5 pt
  * no internal figure title (caption lives in the manuscript)
  * panels:  a network · b masked dispatch · c training · d XAI audit
Outputs: results/fig1_architecture.{pdf,svg,png} and results/architecture_diagram.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, Rectangle, Circle, Polygon)

HERE = Path(__file__).parent
R = HERE / "results"
R.mkdir(exist_ok=True)

# ---------------- Nature (NPG) palette ----------------
INK   = "#1A1A1A"
GREY  = "#6E7680"
BLUE, BLUE_T  = "#3C5488", "#D5DCEA"   # structure / backbone
RED,  RED_T   = "#E64B35", "#F8D9D4"   # risk / mask
GREEN, GREEN_T= "#00A087", "#D2ECE7"   # chosen action / positive
CYAN,  CYAN_T = "#4DBBD5", "#DCEEF4"   # value stream
PURP,  PURP_T = "#8491B4", "#E4E1EC"   # advantage stream
ORNG,  ORNG_T = "#F39B7F", "#FBE9E0"   # reward / audit

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,          # editable text in PDF (journal requirement)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

FIG_W, FIG_H = 183 / 25.4, 133 / 25.4      # 183 mm x 133 mm

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.05], height_ratios=[1, 1],
                      left=0.022, right=0.982, top=0.965, bottom=0.03,
                      wspace=0.10, hspace=0.16)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])
for ax in (axA, axB, axC, axD):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def panel_letter(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=INK, va="bottom", ha="left")


def ptitle(ax, text):
    ax.text(0.0, 1.012, text, transform=ax.transAxes, fontsize=6.8,
            fontweight="bold", color=INK, va="bottom", ha="left")


def arr(ax, x1, y1, x2, y2, color=INK, lw=0.7, ls="-", ms=6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                 shrinkA=0.5, shrinkB=0.5, zorder=6))


def box(ax, x, y, w, h, fc, ec, lw=0.8, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=3))


def blabel(ax, x, y, text, fs=6.2, color=INK, ha="center", va="center",
           style="normal", weight="normal"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va,
            style=style, fontweight=weight, zorder=8)


# ================================================================= #
# PANEL a — network architecture (isometric slabs, flat tints)      #
# ================================================================= #
panel_letter(axA, "a")
ptitle(axA, "Dueling Double-DQN state–action value network")

DX, DY = 2.4, 1.2
BASE = 26


def slab(ax, x, w, h, tint, edge):
    y = BASE
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                         closed=True, fc=tint, ec=edge, lw=0.9, zorder=4))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h),
                          (x + w + DX, y + h + DY), (x + DX, y + h + DY)],
                         closed=True, fc=tint, ec=edge, lw=0.6, alpha=0.55, zorder=3))
    ax.add_patch(Polygon([(x + w, y), (x + w + DX, y + DY),
                          (x + w + DX, y + h + DY), (x + w, y + h)],
                         closed=True, fc=tint, ec=edge, lw=0.6, alpha=0.4, zorder=3))


SW = 8
# input: stacked segments 5 (task) + 40 (VM)
ix, iw = 5, 6.2
box(axA, ix, BASE, iw, 2.8, ORNG_T, ORNG)             # 5 task features
box(axA, ix, BASE + 3.2, iw, 18.5, BLUE_T, BLUE)      # 40 VM features
blabel(axA, ix + iw / 2, BASE + 26.6, "45", 7, INK, weight="bold")
blabel(axA, ix + iw / 2, BASE - 3.6, "state $s_t$", 6, INK)
blabel(axA, ix + iw / 2, BASE - 8.6, "5 task + 40 VM", 5.2, GREY, style="italic")

# backbone
box(axA, 20.5, BASE - 2.5, 22.5, 41.5, "#F6F8FA", GREY, lw=0.5, ls=(0, (3, 2.2)))
blabel(axA, 31.7, BASE + 41.8, "shared backbone", 5.8, GREY, style="italic")
slab(axA, 21.5, SW, 33, BLUE_T, BLUE)      # 128
blabel(axA, 21.5 + SW / 2, BASE + 36.6, "128", 6.5, INK, weight="bold")
slab(axA, 33.5, SW, 23, BLUE_T, BLUE)      # 64
blabel(axA, 33.5 + SW / 2, BASE + 26.6, "64", 6.5, INK, weight="bold")
arr(axA, ix + iw + 0.5, BASE + 10, 20.9, BASE + 14)
arr(axA, 21.5 + SW + 0.6, BASE + 16, 32.9, BASE + 11)

# streams
SX = 49.5
slab(axA, SX, SW, 13, CYAN_T, CYAN)                     # V hidden
blabel(axA, SX + SW / 2, BASE + 32.2, "32", 6.2, INK, weight="bold")
slab(axA, SX, SW - 3.2, 13, PURP_T, PURP, )            # A hidden
blabel(axA, SX + (SW - 3.2) / 2, BASE + 15.6, "32", 6.2, INK, weight="bold")
arr(axA, 33.5 + SW + 0.6, BASE + 12, SX - 0.6, BASE + 30)
arr(axA, 33.5 + SW + 0.6, BASE + 10, SX - 0.6, BASE + 8.5)

# heads
vd = Circle((SX + SW + 7, BASE + 30), 1.6, fc=CYAN, ec=CYAN, zorder=6)
axA.add_patch(vd)
blabel(axA, SX + SW + 7, BASE + 34.4, "$V(s)$", 6.4, "#2E7E93")
arr(axA, SX + SW, BASE + 30, SX + SW + 5.2, BASE + 30)
box(axA, SX + SW, BASE + 3.2, SW, 4.6, PURP_T, PURP)    # 8 advantages
blabel(axA, SX + SW + SW / 2, BASE + 10.2, "8", 6, INK, weight="bold")
blabel(axA, SX + SW + SW / 2, BASE - 3.4, "$A(s,\\cdot)$", 6.2, PURP)
arr(axA, SX + (SW - 3.2) / 2, BASE + 5.5, SX + SW + 0.6, BASE + 6.5)

# aggregation
AGX = 78
axA.add_patch(Circle((AGX, BASE + 19), 2.8, fc="white", ec=INK, lw=0.9, zorder=7))
blabel(axA, AGX, BASE + 19, "$\\Sigma$", 8.5)
arr(axA, SX + SW + 8.8, BASE + 30, AGX - 3.1, BASE + 19)
arr(axA, SX + SW + SW + 0.6, BASE + 5.5, AGX - 3.1, BASE + 19)
blabel(axA, AGX + 0.5, BASE + 25.8, "$Q(s,a)=V+(A-\\bar{A})$", 6.4, INK)
arr(axA, AGX + 3.1, BASE + 19, 92.5, BASE + 19)
blabel(axA, 87, BASE + 22.2, "argmax", 5.4, GREY, style="italic")

# output Q-vector mini-rows
for i in range(4):
    yy = BASE + 26 - i * 5.2
    box(axA, 93, yy, 6, 1.7, GREEN_T if i == 1 else "#F1F3F5",
        GREEN if i == 1 else GREY, lw=0.6)
blabel(axA, 96, BASE + 33.8, "$Q$", 6.4, INK, weight="bold")
blabel(axA, 96, BASE + 2.6, "8 VMs", 5.6, GREY)
blabel(axA, SX + SW + SW / 2, BASE + 34.5, "value stream", 5.6, "#2E7E93",
       style="italic")
blabel(axA, SX + 3.5, BASE - 9.4, "advantage stream", 5.6, PURP, style="italic")

axA.set_xlim(0, 102); axA.set_ylim(8, 76)

# ================================================================= #
# PANEL b — masked dispatch (quantitative example)                  #
# ================================================================= #
panel_letter(axB, "b")
ptitle(axB, "Safety-masked dispatch (example state)")

qv  = [3.1, 2.7, 3.3, 4.2, 2.9, 4.7, 3.2, 2.8]     # VM5 highest but saturated
masked_i, chosen_i = 5, 3
bx0, bw_, gap = 8, 5.2, 2.4
ymax = 30
for i, q in enumerate(qv):
    x = bx0 + i * (bw_ + gap)
    h = (q - 2.0) / (5.0 - 2.0) * ymax
    if i == masked_i:
        axB.add_patch(Rectangle((x, 12), bw_, h, fc="none", ec=RED, lw=0.9,
                                hatch="///", zorder=4))
        blabel(axB, x + bw_ / 2, 12 + h + 2.6, "masked", 5.4, RED)
    elif i == chosen_i:
        axB.add_patch(Rectangle((x, 12), bw_, h, fc=GREEN, ec=GREEN, lw=0.7, zorder=4))
        blabel(axB, x + bw_ / 2, 12 + h + 2.6, "$a^{*}$", 6.6, GREEN, weight="bold")
    else:
        axB.add_patch(Rectangle((x, 12), bw_, h, fc=BLUE, ec=BLUE, lw=0.6,
                                alpha=0.75, zorder=4))
    blabel(axB, x + bw_ / 2, 8.6, f"VM$_{{{i}}}$", 5.2, GREY)
axB.plot([bx0 - 1.5, bx0 + 8 * (bw_ + gap) - 1], [12, 12], color=INK, lw=0.8)
blabel(axB, 3.4, 28, "$Q(s,\\cdot)$", 6.2, INK, ha="left")
blabel(axB, 58, 58.5, "mask rule", 6, RED, weight="bold", ha="left")
blabel(axB, 58, 53.6, "util$_i$>0.9 $\\Rightarrow$ masked", 5.8, INK, ha="left")
blabel(axB, 58, 47.0, "VM$_5$ saturated:", 5.6, GREY, ha="left")
blabel(axB, 58, 42.6, "excluded despite", 5.6, GREY, ha="left")
blabel(axB, 58, 38.2, "highest $Q$", 5.6, GREY, ha="left")
arr(axB, 57.2, 48.5, bx0 + masked_i * (bw_ + gap) + bw_ + 1.4, 31, color=RED,
    lw=0.7, ls=(0, (2.5, 1.8)))
box(axB, 60, 14, 34, 12, GREEN_T, GREEN)
blabel(axB, 77, 22.5, "$a^{*}=\\mathrm{VM}_3$", 7, "#0B7A66", weight="bold")
blabel(axB, 77, 17.8, "dispatch  ·  real time", 5.6, GREY, style="italic")
arr(axB, 56, 20, 59.4, 20, color=GREEN, lw=0.9)
axB.set_xlim(0, 98); axB.set_ylim(2, 62)

# ================================================================= #
# PANEL c — training loop                                           #
# ================================================================= #
panel_letter(axC, "c")
ptitle(axC, "Prioritized replay with double targets")

def mbox(ax, x, y, w, h, txt, fc, ec, fs=6.0, sub=None, sfs=5.2):
    box(ax, x, y, w, h, fc, ec)
    if sub:
        blabel(ax, x + w / 2, y + h * 0.66, txt, fs, INK, weight="bold")
        blabel(ax, x + w / 2, y + h * 0.27, sub, sfs, GREY, style="italic")
    else:
        blabel(ax, x + w / 2, y + h / 2, txt, fs, INK, weight="bold")

mbox(axC, 2.5, 56, 16, 14, "environment", "#F6F8FA", GREY,
     sub="Borg · parallel VMs")
mbox(axC, 2.5, 30, 16, 14, "reward", ORNG_T, ORNG,
     sub="QoS $\\gg$ cost · overload")
arr(axC, 10.5, 44.4, 10.5, 55.4, lw=0.7)

# SumTree
def sumtree(ax, x0, y0, ec):
    lv = [(1, 3.4), (2, 2.2), (4, 1.6)]
    xs_l, ys_l, yy = [], [], y0
    for li, (n, r) in enumerate(lv):
        xs = [x0] if n == 1 else [x0 - 7.5 + i * (15 / (n - 1)) for i in range(n)]
        for xx in xs:
            ax.add_patch(Circle((xx, yy), r * (1.25 if li == 0 else 1),
                                fc="white", ec=ec, lw=0.8, zorder=6))
        xs_l.append(xs); ys_l.append(yy); yy -= 7.2
    for d in range(2):
        for i, p in enumerate(xs_l[d]):
            for c in xs_l[d + 1][2 * i: 2 * i + 2]:
                ax.plot([p, c], [ys_l[d], ys_l[d + 1]], color=ec, lw=0.6, zorder=5)

blabel(axC, 34, 76.5, "SumTree", 6, "#2E7E93", weight="bold")
sumtree(axC, 34, 71, CYAN)
blabel(axC, 34, 48.5, "$P(i)\\!\\propto\\!|\\delta_i|^{\\alpha}$, $\\beta\\!\\to\\!1$, 50k",
       5.4, GREY, style="italic")
arr(axC, 18.9, 63, 24.5, 64.5, lw=0.7)
blabel(axC, 24.0, 72.5, "$(s,a,r,s')$", 5.4, GREY)

mbox(axC, 47, 56, 22, 14, "double target", BLUE_T, BLUE,
     sub="$y\\!=\\!r\\!+\\!\\gamma Q^{-}\\!(s',\\,a^{*})$")
arr(axC, 43.6, 64, 46.4, 63, lw=0.7, color=CYAN)
mbox(axC, 47, 30, 22, 14, "Adam step", PURP_T, PURP,
     sub="IS-weighted MSE")
arr(axC, 58, 55.4, 58, 44.6, lw=0.7)
mbox(axC, 74, 43, 22, 14, "soft sync", PURP_T, PURP, sub="$\\tau\\!=\\!0.005$")
arr(axC, 69.4, 37, 73.4, 43.5, lw=0.7, ls=(0, (2.5, 1.8)), color=PURP)
# loop along bottom
axC.plot([85, 85], [42.6, 10], color=PURP, lw=0.7, ls=(0, (2.5, 1.8)), zorder=2)
axC.plot([85, 10.5], [10, 10], color=PURP, lw=0.7, ls=(0, (2.5, 1.8)), zorder=2)
arr(axC, 10.5, 10, 10.5, 29.4, lw=0.7, ls=(0, (2.5, 1.8)), color=PURP)
blabel(axC, 48, 5.2, "$\\theta$ and $\\theta^{-}$ updates (300 episodes)",
       5.4, PURP, style="italic")
axC.set_xlim(0, 100); axC.set_ylim(0, 84)

# ================================================================= #
# PANEL d — XAI + trust audit (quantitative)                        #
# ================================================================= #
panel_letter(axD, "d")
ptitle(axD, "Attribution methods, audited")

mbox(axD, 1.5, 42, 15, 15, "decision\n$Q(s,a^{*})$", BLUE_T, BLUE, fs=5.8)

rows = [                     # name, deletion AOPC, family
    ("Occlusion",        1.129, PURP),
    ("KernelSHAP",       0.786, CYAN),
    ("Grad × Input",     0.102, BLUE),
    ("Int. gradients",   0.016, BLUE),
]
ys_r = [70, 55, 40, 25]
BARX, BARMAXW = 55.0, 28
for (name, val, col), yy in zip(rows, ys_r):
    blabel(axD, 27.5, yy, name, 5.8, INK, ha="left")
    box(axD, BARX, yy - 1.5, BARMAXW * (val / 1.2), 3.0, col, col, lw=0.5)
    blabel(axD, BARX + BARMAXW * (val / 1.2) + 1.6, yy, f"{val:.2f}", 5.2, GREY,
           ha="left")
    arr(axD, 17.0, 49.5, 25.5, yy, lw=0.55, ms=5)
box(axD, BARX, 8.5, BARMAXW, 0.9, "#EDF0F3", GREY, lw=0.5)
blabel(axD, BARX, 14.5, "deletion AOPC (faithfulness, higher $=$ better)",
       5.2, GREY, style="italic", ha="left")

mbox(axD, 62, 36, 36, 24, "6 trust metrics", ORNG_T, ORNG, fs=6.2,
     sub="ins/del AOPC · top-$k$ fidelity\nconsistency · infidelity · stability")
axD.plot([57.5, 57.5], [23, 73], color=GREY, lw=0.7, zorder=2)
for yy in ys_r:
    arr(axD, 56.2, yy, 57.2, yy, lw=0.55, ms=5, color=GREY)
arr(axD, 58, 48, 61.4, 48, lw=0.8, ms=7, color=GREY)
blabel(axD, 50, 4.5,
       "latency: Grad×In 0.16 ms · Occ 3.3 ms · IG 5.3 ms · SHAP 53.9 ms",
       5.2, GREY, style="italic")
axD.set_xlim(0, 100); axD.set_ylim(0, 82)

# ---------------- save all formats ----------------
for ext in ("pdf", "svg"):
    fig.savefig(R / f"fig1_architecture.{ext}", bbox_inches="tight",
                facecolor="white")
fig.savefig(R / "fig1_architecture.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(R / "architecture_diagram.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("saved: fig1_architecture.pdf / .svg / .png (+ architecture_diagram.png)")
