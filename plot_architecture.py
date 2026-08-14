"""
plot_architecture.py — Journal-style architecture figure (Nature/NeurIPS look):
isometric layer slabs, muted palette, thin arrows, lettered panels (a)(b)(c).
-> results/architecture_diagram.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
OUT = HERE / "results" / "architecture_diagram.png"

# ---- journal palette (muted pastels, dark inks) ----
INK      = "#1A1A1A"
MUTED    = "#5B6770"
SLATE_F  = "#D9E2EC"   # backbone slabs
SLATE_E  = "#425466"
TEAL_F   = "#CFE4E6"; TEAL_E  = "#31707A"   # value stream
PURP_F   = "#E3DBEC"; PURP_E  = "#6A5486"   # advantage stream
ORNG_F   = "#F7E0D2"; ORNG_E  = "#B35F3C"   # mask
GRN_F    = "#D9EBD4"; GRN_E   = "#4E7A45"   # chosen action
GREY_F   = "#EFF1F3"; GREY_E  = "#8A97A0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
})

fig = plt.figure(figsize=(16, 9.0), facecolor="white")

gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1], height_ratios=[1, 1],
                      left=0.03, right=0.985, top=0.90, bottom=0.045,
                      wspace=0.13, hspace=0.24)
axA = fig.add_subplot(gs[:, 0])   # (a) network — full height left
axB = fig.add_subplot(gs[0, 1])   # (b) training
axC = fig.add_subplot(gs[1, 1])   # (c) XAI + trust

fig.suptitle("RL-MOTS-XAI v2: explainable Dueling Double-DQN scheduler for cloud resource allocation",
             fontsize=15.5, fontweight="bold", color=INK, y=0.965)
fig.text(0.5, 0.925,
         "Trained on Google Borg production traces (328 MB, 405k events); every dispatch decision is explained by four attribution methods and audited by six trust metrics.",
         ha="center", fontsize=10, color=MUTED, style="italic")

for ax in (axA, axB, axC):
    ax.axis("off")


def panel(ax, letter, title):
    ax.text(0.01, 0.985, f"({letter})", transform=ax.transAxes,
            fontsize=15, fontweight="bold", color=INK, va="top")
    ax.text(0.045, 0.985, title, transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", color=INK, va="top")


# ------------------------------------------------------------------ #
# (a) NETWORK — isometric slabs                                       #
# ------------------------------------------------------------------ #
panel(axA, "a", "Dueling Double-DQN with state-derived safety mask")

DX, DY = 4.2, 2.2   # isometric depth offsets


def slab(ax, x, y, w, h, fc, ec, depth=(DX, DY)):
    """Pseudo-3D slab: front face + top + side."""
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                         closed=True, fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h),
                          (x + w + depth[0], y + h + depth[1]), (x + depth[0], y + h + depth[1])],
                         closed=True, fc=fc, ec=ec, lw=0.9, alpha=0.75, zorder=2))
    ax.add_patch(Polygon([(x + w, y), (x + w + depth[0], y + depth[1]),
                          (x + w + depth[0], y + h + depth[1]), (x + w, y + h)],
                         closed=True, fc=fc, ec=ec, lw=0.9, alpha=0.55, zorder=2))


def tarrow(ax, x1, y1, x2, y2, color=INK, lw=1.1, ls="-", ms=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                 shrinkA=1, shrinkB=1, zorder=5))


BASE = 30            # baseline y of slab bottoms
SW = 6.4             # slab width
xs = [4, 16, 28]     # input, fc1, fc2 x positions
hs = [16, 34, 22]    # 45, 128, 64 (heights ~ proportional)
dims = ["45", "128", "64"]
labs = ["input\n$s_t$", "", ""]

# grouped background: backbone
axA.add_patch(Rectangle((14.2, BASE - 4), 22.5, 46, fc="#F7F9FB", ec=GREY_E,
                        lw=0.8, ls=(0, (4, 3)), zorder=1))
axA.text(25.4, BASE + 44.5, "shared backbone", ha="center", fontsize=9,
         color=MUTED, style="italic")

slab(axA, xs[0], BASE, SW, hs[0], GREY_F, SLATE_E)
slab(axA, xs[1], BASE, SW, hs[1], SLATE_F, SLATE_E)
slab(axA, xs[2], BASE, SW, hs[2], SLATE_F, SLATE_E)
for x, h, d in zip(xs, hs, dims):
    axA.text(x + SW / 2, BASE + h + 3.4, d, ha="center", fontsize=10,
             color=INK, fontweight="bold")
axA.text(xs[0] + SW / 2, BASE - 3.2, "$s_t\\in\\mathbb{R}^{45}$", ha="center",
         fontsize=9, color=INK)
axA.text(xs[0] + SW / 2, BASE - 7.4, "5 task + 40 VM features", ha="center",
         fontsize=8, color=MUTED, style="italic")

# ---- streams ----
VSX, ASX = 42, 42
VY, AY = BASE + 22, BASE - 2
slab(axA, VSX, VY, SW, 11, TEAL_F, TEAL_E)     # V hidden 32
slab(axA, ASX, AY, SW, 11, PURP_F, PURP_E)     # A hidden 32
axA.text(VSX + SW / 2, VY + 15.5, "32", ha="center", fontsize=9.5, fontweight="bold", color=INK)
axA.text(ASX + SW / 2, AY + 15.5, "32", ha="center", fontsize=9.5, fontweight="bold", color=INK)
axA.text(VSX + SW + 1.4, VY + 5.5, "value stream", fontsize=9, color=TEAL_E, style="italic", va="center")
axA.text(ASX + SW + 1.4, AY + 5.5, "advantage stream", fontsize=9, color=PURP_E, style="italic", va="center")

# heads: V -> scalar circle, A -> 8-slab
tarrow(axA, xs[2] + SW + DX * 0.4, BASE + 11, VSX - 1.2, VY + 5.5)
tarrow(axA, xs[2] + SW + DX * 0.4, BASE + 11, ASX - 1.2, AY + 5.5)

vdot = Circle((VSX + SW + 9, VY + 5.5), 1.7, fc=TEAL_E, ec=TEAL_E, zorder=6)
axA.add_patch(vdot)
axA.text(VSX + SW + 9, VY + 9.6, "$V(s)$", ha="center", fontsize=10, color=TEAL_E)
tarrow(axA, VSX + SW, VY + 5.5, VSX + SW + 7.0, VY + 5.5)

slab(axA, ASX + 2.5, AY - 4.5, SW + 4, 5.2, PURP_F, PURP_E)
axA.text(ASX + 2.5 + (SW + 4) / 2, AY + 3.4, "8", ha="center", fontsize=9, fontweight="bold")
axA.text(ASX + 2.5 + (SW + 4) / 2, AY - 7.6, "$A(s,\\cdot)$", ha="center", fontsize=9, color=PURP_E)
tarrow(axA, ASX + SW / 2, AY, ASX + 2.5 + (SW + 4) / 2, AY - 1.6)

# ---- aggregation ----
AGX = 68
agg = Circle((AGX, BASE + 11), 3.1, fc="white", ec=INK, lw=1.4, zorder=6)
axA.add_patch(agg)
axA.text(AGX, BASE + 11, "$\\Sigma$", ha="center", va="center", fontsize=13, zorder=7)
tarrow(axA, VSX + SW + 10.8, VY + 5.5, AGX - 3.4, BASE + 11)
tarrow(axA, ASX + 2.5 + (SW + 4), AY - 1.9, AGX - 3.4, BASE + 11)
axA.text(AGX, BASE + 17.2, "$Q=V+(A-\\bar{A})$", ha="center", fontsize=10.5, color=INK)

# ---- safety mask gate ----
MGX = 79
for yy in (BASE + 5, BASE + 17):
    axA.add_patch(Rectangle((MGX, yy), 1.5, 3.4, fc=ORNG_E, ec=ORNG_E, zorder=6))
tarrow(axA, AGX + 3.4, BASE + 11, MGX - 1.2, BASE + 11)
axA.text(MGX + 1.0, BASE + 24.5, "safety mask", ha="center", fontsize=9, color=ORNG_E, fontweight="bold")
axA.text(MGX + 1.0, BASE + 21.4, "util > 0.9 $\\Rightarrow$ blocked", ha="center",
         fontsize=8, color=MUTED, style="italic")

# ---- 8 candidate actions ----
OX = 88
ys8 = [BASE + 20 - i * 4.6 for i in range(8)]
for i, yy in enumerate(ys8):
    axA.add_patch(Circle((OX, yy), 1.15, fc="white", ec=SLATE_E, lw=1.1, zorder=6))
    axA.text(OX + 2.2, yy, f"VM$_{{{i}}}$", fontsize=7.5, color=MUTED, va="center")
tarrow(axA, MGX + 2.8, BASE + 11, OX - 1.6, BASE + 11, lw=1.0)
# chosen action
chosen = ys8[3]
axA.add_patch(Circle((OX, chosen), 1.5, fc=GRN_E, ec=GRN_E, zorder=7))
tarrow(axA, OX + 7.4, chosen, OX + 12.4, chosen, color=GRN_E, lw=1.8, ms=11)
axA.text(OX + 13.4, chosen, "$a^{*}$: dispatch\nto VM$_3$", fontsize=9,
         color=GRN_E, va="center", fontweight="bold")

# normalization note between input and fc1
axA.text(13.6, BASE + 42.5, "norm", fontsize=8, color=MUTED, ha="center", style="italic")
tarrow(axA, xs[0] + SW + 0.4, BASE + 8, xs[1] - 0.4, BASE + 17, lw=0.9)
tarrow(axA, xs[1] + SW + 0.4, BASE + 17, xs[2] - 0.4, BASE + 11, lw=0.9)

# ablation footnote inside (a)
axA.text(50, 6.5,
         "Ablation-validated: removing the dueling streams or the safety mask each raises cost by ~35 % (5 seeds, Wilcoxon $p<0.001$).",
         ha="center", fontsize=9, color=MUTED, style="italic")

axA.set_xlim(0, 114); axA.set_ylim(0, 92)

# ------------------------------------------------------------------ #
# (b) TRAINING — SumTree PER + double target                          #
# ------------------------------------------------------------------ #
panel(axB, "b", "Training: prioritized replay + double target")

bx = axB
bx.set_xlim(0, 100); bx.set_ylim(0, 100)

def mbox(ax, x, y, w, h, text, fc, ec, fs=8.6, sub=None, sub_fs=7.4):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=sub_fs, color=MUTED, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold")

# environment & reward
mbox(bx, 2, 62, 20, 13, "environment step", GREY_F, GREY_E,
     sub="Borg task stream · parallel VMs")
mbox(bx, 2, 34, 22, 13, "reward", ORNG_F, ORNG_E,
     sub="QoS $\\gg$ cost · overload · shaping")

# mini SumTree
def sumtree(ax, x0, y0):
    levels = [(1, 5.2), (2, 3.4), (4, 2.4)]   # (n_nodes, radius)
    xs_levels, ys_levels = [], []
    yy = y0
    for li, (n, r) in enumerate(levels):
        if n == 1:
            xs_ = [x0]
        else:
            xs_ = [x0 - 9 + i * (18 / (n - 1)) for i in range(n)]
        for xx in xs_:
            ax.add_patch(Circle((xx, yy), r * (1.3 if li == 0 else 1.0),
                                fc="white", ec=TEAL_E, lw=1.2, zorder=6))
        xs_levels.append(xs_); ys_levels.append(yy)
        yy -= 9.5
    for d in range(len(levels) - 1):
        for i, p in enumerate(xs_levels[d]):
            for c in xs_levels[d + 1][2 * i: 2 * i + 2]:
                ax.plot([p, c], [ys_levels[d], ys_levels[d + 1]],
                        color=TEAL_E, lw=0.8, zorder=5)

bx.text(30.5, 88, "SumTree", fontsize=9.5, color=TEAL_E, fontweight="bold", ha="center")
sumtree(bx, 30.5, 82)

mbox(bx, 46, 62, 24, 13, "prioritized replay", TEAL_F, TEAL_E,
     sub="$P(i)\\propto|\\delta_i|^{\\alpha}$, IS $\\beta\\!\\to\\!1$")
mbox(bx, 46, 34, 24, 13, "double target", SLATE_F, SLATE_E,
     sub="$y=r+\\gamma Q^{-}(s',\\,a^{*})$")

mbox(bx, 75, 48, 24, 13, "Adam · soft sync", PURP_F, PURP_E,
     sub="$\\tau\\!=\\!0.005$")

tarrow(bx, 22.2, 68.5, 45.8, 68.5)
tarrow(bx, 58, 61.8, 58, 47.2)
tarrow(bx, 22.2, 40.5, 45.8, 40.5)
tarrow(bx, 70.2, 40.5, 87, 47.8, color=PURP_E, ls="--", lw=1.0)
tarrow(bx, 87, 61.2, 87, 80.5, color=PURP_E, ls="--", lw=1.0)
tarrow(bx, 87, 80.5, 12, 80.5, color=PURP_E, ls="--", lw=1.0)
tarrow(bx, 12, 80.5, 12, 75.2, color=PURP_E, ls="--", lw=1.0)
bx.text(50, 83.2, "online $\\theta$ updates", fontsize=8, color=PURP_E,
        style="italic", ha="center")
tarrow(bx, 24.5, 72, 24.5, 78.5, color=GREY_E, ls=":", lw=1.0)
bx.text(26, 76.4, "$(s, a, r, s')$", fontsize=7.6, color=MUTED, ha="left")

# ------------------------------------------------------------------ #
# (c) XAI + TRUST                                                     #
# ------------------------------------------------------------------ #
panel(axC, "c", "Explainability + trust audit of every decision")

cx = axC
cx.set_xlim(0, 100); cx.set_ylim(0, 100)

mbox(cx, 2, 40, 18, 16, "decision\n$Q(s, a^{*})$", SLATE_F, SLATE_E, fs=9)

methods = [("KernelSHAP", "250 coalitions · WLS", TEAL_F, TEAL_E),
           ("Gradient×Input", "exact · 0.15 ms", TEAL_F, TEAL_E),
           ("Occlusion", "windowed $\\Delta Q$", PURP_F, PURP_E),
           ("Integrated Grads", "path integral", PURP_F, PURP_E)]
yy = 72
for name, sub, fc, ec in methods:
    mbox(cx, 30, yy, 27, 12, name, fc, ec, fs=8.8, sub=sub)
    tarrow(cx, 20.2, 48, 29.8, yy + 6, lw=0.9, ms=8)
    yy -= 19.5

mbox(cx, 66, 40, 32, 16, "6 trust metrics", ORNG_F, ORNG_E, fs=9.5,
     sub="deletion/insertion AOPC · top-$k$ fidelity\nconsistency · infidelity · stability")
for yv in (78, 58.5, 39, 19.5):
    tarrow(cx, 57.2, yv, 65.8, 50 if yv != 50 else 50, lw=0.9, ms=8)

cx.text(50, 6,
        "Occlusion & KernelSHAP most faithful (AOPC $+$0.73/$+$0.56); Gradient×Input 340× faster — audited trade-off.",
        ha="center", fontsize=8.4, color=MUTED, style="italic")

fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print(f"saved -> {OUT}")
