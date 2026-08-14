"""
plot_architecture.py — Publication-format architecture figure (Nature style).

183 mm double-column, NPG palette, print-calibrated type, four panels.
Includes an automatic collision checker: every text bounding box is tested
against every arrow segment and every other text box — generation reports
violations so overlaps cannot slip through.

Outputs: results/fig1_architecture.{pdf,svg,png} + results/architecture_diagram.png
"""
from pathlib import Path
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Polygon

HERE = Path(__file__).parent
R = HERE / "results"
R.mkdir(exist_ok=True)

INK, GREY = "#1A1A1A", "#6E7680"
BLUE, BLUE_T  = "#3C5488", "#DDE3EF"
RED,  RED_T   = "#E64B35", "#F8D9D4"
GREEN, GREEN_T= "#00A087", "#D2ECE7"
CYAN,  CYAN_T = "#4DBBD5", "#DDEEF4"
PURP,  PURP_T = "#8491B4", "#E6E3EE"
ORNG,  ORNG_T = "#F39B7F", "#FBE9E0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
})

FIG_W, FIG_H = 183 / 25.4, 133 / 25.4
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.05], height_ratios=[1, 1],
                      left=0.022, right=0.982, top=0.955, bottom=0.035,
                      wspace=0.11, hspace=0.20)
AX = {k: fig.add_subplot(gs[i, j])
      for k, (i, j) in {"a": (0, 0), "b": (0, 1), "c": (1, 0), "d": (1, 1)}.items()}
for ax in AX.values():
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

TEXTS, ARROWS = [], []


def panel_letter(p):
    AX[p].text(-0.028, 1.045, p, transform=AX[p].transAxes, fontsize=9.5,
               fontweight="bold", color=INK, va="bottom", ha="left")


def ptitle(p, text):
    AX[p].text(0.0, 1.015, text, transform=AX[p].transAxes, fontsize=6.8,
               fontweight="bold", color=INK, va="bottom", ha="left")


def txt(p, x, y, s, fs=6.0, color=INK, ha="center", va="center",
        style="normal", weight="normal", rot=0):
    t = AX[p].text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, rotation=rot,
                   style=style, fontweight=weight, zorder=9)
    TEXTS.append((p, t))
    return t


def arr(p, x1, y1, x2, y2, color=INK, lw=0.7, ls="-", ms=6):
    AX[p].add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                    mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                    shrinkA=0.5, shrinkB=0.5, zorder=6))
    ARROWS.append((p, x1, y1, x2, y2))


def line(p, x1, y1, x2, y2, color=GREY, lw=0.7, ls="-"):
    AX[p].plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=2)
    ARROWS.append((p, x1, y1, x2, y2))


def box(p, x, y, w, h, fc, ec, lw=0.8, ls="-", hatch=None):
    AX[p].add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw,
                              linestyle=ls, hatch=hatch, zorder=3))


def mbox(p, x, y, w, h, t, fc, ec, fs=6.0, sub=None, sfs=5.2):
    box(p, x, y, w, h, fc, ec)
    txt(p, x + w / 2, y + h * (0.64 if sub else 0.5), t, fs, INK, weight="bold")
    if sub:
        txt(p, x + w / 2, y + h * 0.27, sub, sfs, GREY, style="italic")


# ================================================================= #
# PANEL a — network                                                 #
# ================================================================= #
panel_letter("a"); ptitle("a", "Dueling Double-DQN state–action value network")
a = AX["a"]
DX, DY = 2.2, 1.1


def slab(x, y, w, h, tint, edge):
    a.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                        closed=True, fc=tint, ec=edge, lw=0.9, zorder=4))
    a.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w + DX, y + h + DY),
                         (x + DX, y + h + DY)], closed=True, fc=tint, ec=edge,
                        lw=0.6, alpha=0.55, zorder=3))
    a.add_patch(Polygon([(x + w, y), (x + w + DX, y + DY),
                         (x + w + DX, y + h + DY), (x + w, y + h)],
                        closed=True, fc=tint, ec=edge, lw=0.6, alpha=0.40, zorder=3))


# ---- input stack (5 task + 40 VM) ----
ix, iw = 4, 6
box("a", ix, 22, iw, 2.6, ORNG_T, ORNG)
box("a", ix, 25.2, iw, 17.0, BLUE_T, BLUE)
txt("a", ix + iw / 2, 46.4, "45", 7, INK, weight="bold")
txt("a", ix + iw / 2, 17.4, "state $s_t$", 6, INK)
txt("a", ix + iw / 2, 12.2, "5 task + 40 VM", 5.2, GREY, style="italic")

# ---- shared backbone ----
box("a", 17.5, 20, 26.5, 37, "#F7F9FB", GREY, lw=0.5, ls=(0, (3, 2.2)))
txt("a", 30.7, 61.4, "shared backbone", 5.8, GREY, style="italic")
slab(19, 22, 7.5, 29, BLUE_T, BLUE)
txt("a", 22.75, 54.4, "128", 6.2, INK, weight="bold")
slab(31.5, 22, 7.5, 21, BLUE_T, BLUE)
txt("a", 35.25, 46.2, "64", 6.2, INK, weight="bold")
arr("a", 10.5, 33, 18.4, 36)
arr("a", 26.9, 38, 30.9, 34)

# ---- streams (vertically separated, big gap) ----
SX, SW = 47, 7.5
slab(SX, 48, SW, 12, CYAN_T, CYAN)                 # V: y 48-60
txt("a", SX + SW / 2, 63.2, "32", 6.0, INK, weight="bold")
txt("a", SX + SW / 2, 68.2, "value stream", 5.4, "#2E7E93", style="italic")
slab(SX, 26, SW - 2.5, 12, PURP_T, PURP)           # A: y 26-38
txt("a", SX + (SW - 2.5) / 2, 41.6, "32", 6.0, INK, weight="bold")
txt("a", SX + (SW - 2.5) / 2, 20.6, "advantage stream", 5.4, PURP, style="italic")
arr("a", 39.4, 36, 46.2, 54)
arr("a", 39.4, 30, 46.2, 32)

# ---- heads ----
a.add_patch(Circle((60.5, 54), 1.6, fc=CYAN, ec=CYAN, zorder=6))
txt("a", 60.5, 59.6, "$V(s)$", 6.2, "#2E7E93")
arr("a", 54.9, 54, 58.6, 54)
box("a", 56.5, 24, 9.0, 4.4, PURP_T, PURP)
txt("a", 61.0, 31.4, "8", 5.8, INK, weight="bold")
txt("a", 61.0, 18.6, "$A(s,\\cdot)$", 6.0, PURP)
arr("a", 49.75, 26, 56.2, 27)

# ---- aggregation ----
AGX, AGY = 76, 44
a.add_patch(Circle((AGX, AGY), 2.8, fc="white", ec=INK, lw=0.9, zorder=7))
txt("a", AGX, AGY, "$\\Sigma$", 8.5)
txt("a", 72, 66.5, "$Q=V+(A-\\bar{A})$", 6.2, INK)
arr("a", 62.3, 54, 72.9, 45.5)
arr("a", 65.7, 26.5, 72.9, 42.5)
arr("a", 79.2, 44, 87.8, 44)
txt("a", 83.5, 39.0, "argmax", 5.2, GREY, style="italic")

# ---- output Q rows ----
for i in range(4):
    box("a", 88.5, 52 - i * 7.5, 6.5, 2.0,
        GREEN_T if i == 0 else "#F1F3F5", GREEN if i == 0 else GREY, lw=0.6)
txt("a", 91.7, 58.2, "$Q$", 6.2, INK, weight="bold")
txt("a", 91.7, 28.4, "8 VMs", 5.4, GREY)
AX["a"].set_xlim(0, 100); AX["a"].set_ylim(4, 74)

# ================================================================= #
# PANEL b — masked dispatch                                         #
# ================================================================= #
panel_letter("b"); ptitle("b", "Safety-masked dispatch (example state)")
qv = [3.1, 2.7, 3.3, 4.2, 2.9, 4.7, 3.2, 2.8]
MASKED, CHOSEN = 5, 3
bx0, bw_, gap = 6, 5.0, 2.4
for i, q in enumerate(qv):
    x = bx0 + i * (bw_ + gap)
    h = (q - 2.0) / 3.0 * 28
    if i == MASKED:
        box("b", x, 12, bw_, h, "none", RED, lw=0.9, hatch="///")
        txt("b", x + bw_ / 2, 12 + h + 3.2, "masked", 5.2, RED)
    elif i == CHOSEN:
        box("b", x, 12, bw_, h, GREEN, GREEN, lw=0.6)
        txt("b", x + bw_ / 2, 12 + h + 3.2, "$a^{*}$", 6.4, GREEN, weight="bold")
    else:
        box("b", x, 12, bw_, h, BLUE, BLUE, lw=0.5)
    txt("b", x + bw_ / 2, 7.6, f"VM$_{{{i}}}$", 5.0, GREY)
line("b", bx0 - 1.6, 12, bx0 + 8 * (bw_ + gap) - 1.2, 12, INK, 0.8, "-")
txt("b", 2.6, 28, "$Q(s,\\cdot)$", 6.0, INK, rot=90)

txt("b", 67, 60.5, "mask rule", 5.8, RED, ha="left", weight="bold")
txt("b", 67, 55.0, "util$_i$>0.9 $\\Rightarrow$ masked", 5.6, INK, ha="left")
arr("b", 66.2, 57.5, bx0 + MASKED * (bw_ + gap) + bw_ + 1.0, 33.0,
    color=RED, lw=0.7, ls=(0, (2.5, 1.8)))

box("b", 67, 16, 29, 12, GREEN_T, GREEN)
txt("b", 81.5, 24.3, "$a^{*}=\\mathrm{VM}_3$", 6.6, "#0B7A66", weight="bold")
txt("b", 81.5, 19.3, "dispatch · real time", 5.4, GREY, style="italic")
arr("b", 63.5, 22, 66.4, 22, color=GREEN, lw=0.9, ms=7)
AX["b"].set_xlim(0, 99); AX["b"].set_ylim(2, 68)

# ================================================================= #
# PANEL c — training loop                                           #
# ================================================================= #
panel_letter("c"); ptitle("c", "Prioritized replay with double targets")

mbox("c", 2.5, 55, 16, 14, "environment", "#F7F9FB", GREY, sub="Borg · parallel VMs")
mbox("c", 2.5, 29, 16, 14, "reward", ORNG_T, ORNG, sub="QoS $\\gg$ cost · overload")
arr("c", 10.5, 43.4, 10.5, 54.6)

txt("c", 34, 82.5, "SumTree", 6.0, "#2E7E93", weight="bold")


def sumtree(x0, y0, ec):
    lv = [(1, 3.2), (2, 2.1), (4, 1.5)]
    xs_l, ys_l, yy = [], [], y0
    for li, (n, r) in enumerate(lv):
        xs = [x0] if n == 1 else [x0 - 7.0 + i * (14 / (n - 1)) for i in range(n)]
        for xx in xs:
            AX["c"].add_patch(Circle((xx, yy), r * (1.25 if li == 0 else 1),
                                     fc="white", ec=ec, lw=0.8, zorder=6))
        xs_l.append(xs); ys_l.append(yy); yy -= 6.8
    for d in range(2):
        for i, p in enumerate(xs_l[d]):
            for c in xs_l[d + 1][2 * i: 2 * i + 2]:
                line("c", p, ys_l[d], c, ys_l[d + 1], ec, 0.6, "-")


sumtree(34, 74, CYAN)
txt("c", 34, 52.5, "$P(i)\\!\\propto\\!|\\delta_i|^{\\alpha}$ · $\\beta\\!\\to\\!1$ · 50k",
    5.2, GREY, style="italic")
arr("c", 18.9, 62, 30.2, 71.5)
txt("c", 24.0, 75.5, "$(s,a,r,s')$", 5.2, GREY)

mbox("c", 47, 55, 22, 14, "double target", BLUE_T, BLUE,
     sub="$y\\!=\\!r\\!+\\!\\gamma Q^{-}\\!(s',a^{*})$")
arr("c", 43.5, 63, 46.4, 63, color=CYAN)
mbox("c", 47, 29, 22, 14, "Adam step", PURP_T, PURP, sub="IS-weighted MSE")
arr("c", 58, 54.6, 58, 43.4)
mbox("c", 74, 42, 22, 14, "soft sync", PURP_T, PURP, sub="$\\tau\\!=\\!0.005$")
arr("c", 69.4, 36, 73.4, 42.5, ls=(0, (2.5, 1.8)), color=PURP)
line("c", 85, 41.6, 85, 9.5, PURP, 0.7, (0, (2.5, 1.8)))
line("c", 85, 9.5, 10.5, 9.5, PURP, 0.7, (0, (2.5, 1.8)))
arr("c", 10.5, 9.5, 10.5, 28.6, ls=(0, (2.5, 1.8)), color=PURP)
txt("c", 48, 4.6, "$\\theta$ and $\\theta^{-}$ updates (300 episodes)", 5.2,
    PURP, style="italic")
AX["c"].set_xlim(0, 100); AX["c"].set_ylim(0, 88)

# ================================================================= #
# PANEL d — XAI + trust audit                                       #
# ================================================================= #
panel_letter("d"); ptitle("d", "Attribution methods, audited")

mbox("d", 1.5, 41, 15, 15, "decision\n$Q(s,a^{*})$", BLUE_T, BLUE, fs=5.6)

rows = [("Occlusion", 1.129, PURP), ("KernelSHAP", 0.786, CYAN),
        ("Grad × Input", 0.102, BLUE), ("Int. gradients", 0.016, BLUE)]
ys_r = [69, 54, 39, 24]
BARX, BARMAXW = 52, 26
for (name, val, col), yy in zip(rows, ys_r):
    txt("d", 26, yy, name, 5.8, INK, ha="left")
    box("d", BARX, yy - 1.5, BARMAXW * (val / 1.2), 3.0, col, col, lw=0.5)
    txt("d", BARX + BARMAXW * (val / 1.2) + 1.8, yy, f"{val:.2f}", 5.2,
        GREY, ha="left")
    arr("d", 17.0, 48.5, 24.5, yy, lw=0.55, ms=5)
box("d", BARX, 18.5, BARMAXW, 0.9, "#EDF0F3", GREY, lw=0.5)
txt("d", 10, 13.5, "deletion AOPC (higher $=$ better)",
    5.2, GREY, style="italic", ha="left")

box("d", 58, 8.5, 40, 15, ORNG_T, ORNG)
txt("d", 78, 19.5, "6 trust metrics", 6.0, INK, weight="bold")
txt("d", 78, 13.0, "ins/del AOPC · fidelity\nconsistency · infidelity · stability",
    4.8, GREY, style="italic")
arr("d", 50, 16, 57.4, 16, lw=0.8, ms=7, color=GREY)
txt("d", 50, 3.0, "latency: G×I 0.16 · Occ 3.3 · IG 5.3 · SHAP 53.9 ms",
    5.2, GREY, style="italic")
AX["d"].set_xlim(0, 100); AX["d"].set_ylim(0, 82)


# ================================================================= #
# COLLISION CHECKER                                                 #
# ================================================================= #
def check_collisions():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    violations = []

    bboxes = []
    for p, t in TEXTS:
        bboxes.append((p, t.get_text()[:26].replace("\n", " "),
                       t.get_window_extent(renderer)))

    for (p1, s1, b1), (p2, s2, b2) in itertools.combinations(bboxes, 2):
        if p1 == p2 and b1.overlaps(b2):
            violations.append(f"[{p1}] TEXT×TEXT: '{s1}' vs '{s2}'")

    import numpy as np
    for p, x1, y1, x2, y2 in ARROWS:
        ax_ = AX[p]
        xs = np.linspace(x1, x2, 40); ys = np.linspace(y1, y2, 40)
        pts = ax_.transData.transform(np.column_stack([xs, ys]))
        for tp, s, bb in bboxes:
            if tp != p:
                continue
            inside = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                      (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1))
            if inside.any():
                violations.append(f"[{p}] LINE×TEXT: ({x1:.0f},{y1:.0f})-"
                                  f"({x2:.0f},{y2:.0f}) crosses '{s}'")

    for p, s, bb in bboxes:
        ab = AX[p].get_window_extent(renderer)
        if (bb.x0 < ab.x0 - 2 or bb.x1 > ab.x1 + 2 or
                bb.y0 < ab.y0 - 2 or bb.y1 > ab.y1 + 2):
            violations.append(f"[{p}] OUT-OF-PANEL: '{s}'")

    if violations:
        print(f"COLLISIONS ({len(violations)}):")
        for v in violations:
            print("  ✗", v)
    else:
        print("collision check: CLEAN — 0 violations")
    return violations


check_collisions()

for ext in ("pdf", "svg"):
    fig.savefig(R / f"fig1_architecture.{ext}", bbox_inches="tight", facecolor="white")
fig.savefig(R / "fig1_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(R / "architecture_diagram.png", dpi=300, bbox_inches="tight", facecolor="white")
print("figures saved")
