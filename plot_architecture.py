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

HERE = Path(__file__).parent
OUT = HERE / "results" / "architecture_diagram.png"

# ---- journal palette (muted pastels, dark inks) ----
INK      = "#1A1A1A"
MUTED    = "#5B6770"
SLATE_F  = "#D9E2EC"; SLATE_E = "#425466"
TEAL_F   = "#CFE4E6"; TEAL_E  = "#31707A"
PURP_F   = "#E3DBEC"; PURP_E  = "#6A5486"
ORNG_F   = "#F7E0D2"; ORNG_E  = "#B35F3C"
GRN_F    = "#D9EBD4"; GRN_E   = "#4E7A45"
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
axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 1])

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
    ax.text(0.048, 0.985, title, transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", color=INK, va="top")


def tarrow(ax, x1, y1, x2, y2, color=INK, lw=1.1, ls="-", ms=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                 shrinkA=1, shrinkB=1, zorder=5))


def mbox(ax, x, y, w, h, text, fc, ec, fs=8.6, sub=None, sub_fs=7.4):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, text, ha="center", va="center",
                fontsize=fs, color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=sub_fs, color=MUTED, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, fontweight="bold")


# ================================================================== #
# (a) NETWORK — isometric slabs                                      #
# ================================================================== #
panel(axA, "a", "Dueling Double-DQN with state-derived safety mask")

DX, DY = 4.2, 2.2


def slab(ax, x, y, w, h, fc, ec):
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                         closed=True, fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h),
                          (x + w + DX, y + h + DY), (x + DX, y + h + DY)],
                         closed=True, fc=fc, ec=ec, lw=0.9, alpha=0.75, zorder=2))
    ax.add_patch(Polygon([(x + w, y), (x + w + DX, y + DY),
                          (x + w + DX, y + h + DY), (x + w, y + h)],
                         closed=True, fc=fc, ec=ec, lw=0.9, alpha=0.55, zorder=2))


BASE, SW = 30, 6.4
xs  = [4, 16, 28]
hs  = [16, 34, 22]
dims = ["45", "128", "64"]

axA.add_patch(Rectangle((14.2, BASE - 4), 22.5, 46, fc="#F7F9FB", ec=GREY_E,
                        lw=0.8, ls=(0, (4, 3)), zorder=1))
axA.text(25.4, BASE + 44.8, "shared backbone", ha="center", fontsize=9,
         color=MUTED, style="italic")

slab(axA, xs[0], BASE, SW, hs[0], GREY_F, SLATE_E)
slab(axA, xs[1], BASE, SW, hs[1], SLATE_F, SLATE_E)
slab(axA, xs[2], BASE, SW, hs[2], SLATE_F, SLATE_E)
for x, h, d in zip(xs, hs, dims):
    axA.text(x + SW / 2, BASE + h + 3.6, d, ha="center", fontsize=10,
             color=INK, fontweight="bold")
axA.text(xs[0] + SW / 2, BASE - 3.4, "$s_t\\in\\mathbb{R}^{45}$", ha="center",
         fontsize=9, color=INK)
axA.text(xs[0] + SW / 2, BASE - 8.0, "5 task + 40 VM features", ha="center",
         fontsize=8, color=MUTED, style="italic")
tarrow(axA, xs[0] + SW + 0.4, BASE + 8, xs[1] - 0.4, BASE + 17, lw=0.9)
tarrow(axA, xs[1] + SW + 0.4, BASE + 17, xs[2] - 0.4, BASE + 11, lw=0.9)
axA.text(13.6, BASE + 43.8, "norm", fontsize=8, color=MUTED, ha="center",
         style="italic")

# ---- streams ----
VSX, ASX = 42, 42
VY, AY = BASE + 22, BASE - 2
slab(axA, VSX, VY, SW, 11, TEAL_F, TEAL_E)
slab(axA, ASX, AY, SW, 11, PURP_F, PURP_E)
axA.text(VSX + SW / 2, VY + 15.6, "32", ha="center", fontsize=9.5,
         fontweight="bold", color=INK)
axA.text(ASX + SW / 2, AY + 15.6, "32", ha="center", fontsize=9.5,
         fontweight="bold", color=INK)
axA.text(VSX - 3.5, VY + 21.5, "value stream", fontsize=9, color=TEAL_E,
         style="italic", ha="left")
axA.text(ASX - 3.5, AY + 21.5, "advantage stream", fontsize=9, color=PURP_E,
         style="italic", ha="left")
tarrow(axA, xs[2] + SW + DX * 0.4, BASE + 11, VSX - 1.2, VY + 5.5)
tarrow(axA, xs[2] + SW + DX * 0.4, BASE + 11, ASX - 1.2, AY + 5.5)

vdot = Circle((VSX + SW + 8.5, VY + 5.5), 1.7, fc=TEAL_E, ec=TEAL_E, zorder=6)
axA.add_patch(vdot)
axA.text(VSX + SW + 8.5, VY + 9.8, "$V(s)$", ha="center", fontsize=10, color=TEAL_E)
tarrow(axA, VSX + SW, VY + 5.5, VSX + SW + 6.6, VY + 5.5)

slab(axA, ASX + 2.5, AY - 4.5, SW + 4, 5.2, PURP_F, PURP_E)
axA.text(ASX + 2.5 + (SW + 4) / 2, AY + 3.6, "8", ha="center", fontsize=9,
         fontweight="bold")
axA.text(ASX + 2.5 + (SW + 4) / 2, AY - 8.2, "$A(s,\\cdot)$", ha="center",
         fontsize=9, color=PURP_E)
tarrow(axA, ASX + SW / 2, AY, ASX + 2.5 + (SW + 4) / 2, AY - 1.6)

# ---- aggregation ----
AGX = 68
axA.add_patch(Circle((AGX, BASE + 11), 3.1, fc="white", ec=INK, lw=1.4, zorder=6))
axA.text(AGX, BASE + 11, "$\\Sigma$", ha="center", va="center", fontsize=13, zorder=7)
tarrow(axA, VSX + SW + 10.3, VY + 5.5, AGX - 3.4, BASE + 11)
tarrow(axA, ASX + 2.5 + SW + 4, AY - 1.9, AGX - 3.4, BASE + 11)
axA.text(AGX, BASE + 18.0, "$Q=V+(A-\\bar{A})$", ha="center", fontsize=10.5, color=INK)

# ---- safety mask gate ----
MGX = 79
for yy in (BASE + 6, BASE + 16):
    axA.add_patch(Rectangle((MGX, yy), 1.5, 3.4, fc=ORNG_E, ec=ORNG_E, zorder=6))
tarrow(axA, AGX + 3.4, BASE + 11, MGX - 1.2, BASE + 11)
axA.text(MGX - 4.5, BASE + 28.5, "safety mask", ha="center", fontsize=9,
         color=ORNG_E, fontweight="bold")
axA.text(MGX - 4.5, BASE + 25.2, "util > 0.9 $\\Rightarrow$ blocked", ha="center",
         fontsize=8, color=MUTED, style="italic")

# ---- 8 candidate actions ----
OX = 88
ys8 = [BASE + 20 - i * 4.6 for i in range(8)]
for i, yy in enumerate(ys8):
    axA.add_patch(Circle((OX, yy), 1.15, fc="white", ec=SLATE_E, lw=1.1, zorder=6))
    axA.text(OX + 2.4, yy, f"VM$_{{{i}}}$", fontsize=7.5, color=MUTED, va="center")
tarrow(axA, MGX + 2.8, BASE + 11, OX - 1.6, BASE + 11, lw=1.0)

chosen = ys8[3]
axA.add_patch(Circle((OX, chosen), 1.5, fc=GRN_E, ec=GRN_E, zorder=7))
tarrow(axA, OX + 8.6, chosen, OX + 12.6, chosen, color=GRN_E, lw=1.8, ms=11)
axA.text(OX + 14.0, chosen, "$a^{*}$: dispatch\nto VM$_3$", fontsize=9,
         color=GRN_E, va="center", fontweight="bold")

axA.text(52, 6.0,
         "Ablation-validated: removing the dueling streams or the safety mask each raises cost by ~35 % (5 seeds, Wilcoxon $p<0.001$).",
         ha="center", fontsize=9, color=MUTED, style="italic")
axA.set_xlim(0, 116); axA.set_ylim(0, 92)

# ================================================================== #
# (b) TRAINING — clean left-to-right chain, loop routed below         #
# ================================================================== #
panel(axB, "b", "Training: prioritized replay + double target")
bx = axB
bx.set_xlim(0, 100); bx.set_ylim(0, 100)

mbox(bx, 3, 56, 17, 15, "environment", GREY_F, GREY_E,
     sub="Borg stream ·\nparallel VMs", sub_fs=7.2)
mbox(bx, 3, 28, 17, 15, "reward", ORNG_F, ORNG_E,
     sub="QoS $\\gg$ cost\noverload · shaping", sub_fs=7.2)
tarrow(bx, 11.5, 43.2, 11.5, 55.6, lw=1.0)

# SumTree (drawn)
def sumtree(ax, x0, y_root):
    levels = [(1, 4.6), (2, 3.0), (4, 2.1)]
    xs_l, ys_l = [], []
    yy = y_root
    for li, (n, r) in enumerate(levels):
        xs_ = [x0] if n == 1 else [x0 - 9 + i * (18 / (n - 1)) for i in range(n)]
        for xx in xs_:
            ax.add_patch(Circle((xx, yy), r * (1.3 if li == 0 else 1.0),
                                fc="white", ec=TEAL_E, lw=1.2, zorder=6))
        xs_l.append(xs_); ys_l.append(yy)
        yy -= 9.0
    for d in range(len(levels) - 1):
        for i, p in enumerate(xs_l[d]):
            for c in xs_l[d + 1][2 * i: 2 * i + 2]:
                ax.plot([p, c], [ys_l[d], ys_l[d + 1]], color=TEAL_E, lw=0.8, zorder=5)
    return xs_l, ys_l

sumtree(bx, 40, 76)
bx.text(40, 87, "SumTree replay", fontsize=9.5, color=TEAL_E,
        fontweight="bold", ha="center")
bx.text(40, 47.5, "$P(i)\\propto|\\delta_i|^{\\alpha}$ · IS $\\beta\\to1$ · 50k",
        fontsize=7.8, color=MUTED, ha="center", style="italic")

tarrow(bx, 20.4, 63.5, 29.6, 68.5)
bx.text(25, 74.5, "$(s,a,r,s')$", fontsize=7.8, color=MUTED, ha="center")

mbox(bx, 55, 56, 22, 14, "double target", SLATE_F, SLATE_E,
     sub="$y=r+\\gamma Q^{-}(s',\\, a^{*})$", sub_fs=7.8)
tarrow(bx, 49.4, 65, 54.6, 63.5, lw=1.0)
bx.text(52, 68.8, "sample", fontsize=7.5, color=MUTED, ha="center", style="italic")

mbox(bx, 55, 28, 22, 14, "Adam step", PURP_F, PURP_E,
     sub="weighted MSE · clip", sub_fs=7.6)
tarrow(bx, 66, 55.6, 66, 42.4, lw=1.0)

mbox(bx, 83, 42, 15, 14, "soft sync", PURP_F, PURP_E,
     sub="$\\tau\\!=\\!0.005$", sub_fs=7.8)
tarrow(bx, 77.4, 35, 82.6, 42, color=PURP_E, lw=1.0, ls="--")

# dashed loop routed cleanly below everything
bx.plot([90.5, 90.5], [41.8, 12], color=PURP_E, lw=1.0, ls="--", zorder=2)
bx.plot([90.5, 11.5], [12, 12], color=PURP_E, lw=1.0, ls="--", zorder=2)
tarrow(bx, 11.5, 12, 11.5, 27.4, color=PURP_E, lw=1.0, ls="--")
bx.text(50.5, 8.0, "online $\\theta$ + target $\\theta^{-}$ updates", fontsize=8,
        color=PURP_E, style="italic", ha="center")

# ================================================================== #
# (c) XAI + TRUST — fan-out then a collection rail (no crossings)     #
# ================================================================== #
panel(axC, "c", "Explainability + trust audit of every decision")
cx = axC
cx.set_xlim(0, 100); cx.set_ylim(0, 100)

mbox(cx, 2, 42, 16, 16, "decision\n$Q(s, a^{*})$", SLATE_F, SLATE_E, fs=9)

methods = [("KernelSHAP", "250 coalitions · WLS", TEAL_F, TEAL_E),
           ("Gradient×Input", "exact · 0.15 ms", TEAL_F, TEAL_E),
           ("Occlusion", "windowed $\\Delta Q$", PURP_F, PURP_E),
           ("Integrated Grads", "path integral", PURP_F, PURP_E)]
ys_m = [78, 60.5, 43, 25.5]
for (name, sub, fc, ec), yy in zip(methods, ys_m):
    mbox(cx, 28, yy, 26, 11.5, name, fc, ec, fs=8.8, sub=sub)
    tarrow(cx, 18.4, 50, 27.6, yy + 5.75, lw=0.9, ms=8)

# collection rail
cx.plot([60, 60], [31, 84], color=GREY_E, lw=1.1, zorder=2)
for yy in ys_m:
    tarrow(cx, 54.2, yy + 5.75, 59.5, yy + 5.75, color=GREY_E, lw=0.9, ms=8)

mbox(cx, 66, 40, 32, 17, "6 trust metrics", ORNG_F, ORNG_E, fs=9.5,
     sub="deletion / insertion AOPC · top-$k$ fidelity\nconsistency · infidelity · stability")
tarrow(cx, 60.5, 50, 65.6, 50, color=GREY_E, lw=1.2, ms=10)

cx.text(50, 8,
        "Occlusion & KernelSHAP most faithful (AOPC $+$0.73 / $+$0.56); Gradient×Input 340× faster — an audited trade-off.",
        ha="center", fontsize=8.2, color=MUTED, style="italic")

fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print(f"saved -> {OUT}")
