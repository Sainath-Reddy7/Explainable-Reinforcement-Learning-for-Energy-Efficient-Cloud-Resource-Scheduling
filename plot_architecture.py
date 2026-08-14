"""
plot_architecture.py — Publication-quality architecture diagram of the
Dueling Double-DQN + safety mask + XAI stack.  -> results/architecture_diagram.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
OUT = HERE / "results" / "architecture_diagram.png"

NAVY, BLUE, TEAL = "#0F2847", "#0066CC", "#17A2B8"
ORANGE, GREEN, GREY, PURPLE = "#E27D60", "#2D9B6B", "#7A8794", "#7D6B91"
LIGHT = "#EBF3FA"

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 108)
ax.set_ylim(0, 104)
ax.axis("off")


def box(x, y, w, h, text, fc, ec=None, tc="white", fs=11, bold=True,
        sub=None, sub_fs=9, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.55,rounding_size=1.3",
                 fc=fc, ec=ec or fc, lw=lw))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color=tc)
        ax.text(x + w / 2, y + h * 0.29, sub, ha="center", va="center",
                fontsize=sub_fs, color=tc, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color=tc)


def arrow(x1, y1, x2, y2, color=NAVY, lw=2.6, ls="-", alpha=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=24, color=color, lw=lw, linestyle=ls,
                 alpha=alpha, shrinkA=3, shrinkB=3))


def lane_label(x, y, text, color=NAVY):
    ax.text(x, y, text, rotation=90, va="center", ha="center",
            fontsize=9.5, fontweight="bold", color=color, alpha=0.55)


# ============================ TITLE ============================
ax.text(54, 99, "RL-MOTS-XAI v2 — Dueling Double-DQN Cloud Scheduler",
        ha="center", fontsize=18, fontweight="bold", color=NAVY)
ax.text(54, 95.2, "Pure-NumPy implementation  ·  trained on real Google Borg traces  ·  "
        "every decision audited by 4 XAI methods",
        ha="center", fontsize=11, color=GREY, style="italic")

# ==================== ROW A: FORWARD PATH =====================
yA, hA = 68, 14
lane_label(2.6, yA + hA / 2, "FORWARD  PATH  (decision)")

box(6,  yA, 13, hA, "State  $s_t\\in\\mathbb{R}^{45}$", BLUE, sub="5 task + 40 VM feats", fs=11.5)
box(22, yA, 11, hA, "Input Norm", GREY, sub="running μ, σ", fs=11)
box(36, yA, 14, hA, "Shared Backbone", NAVY, sub="128 → 64 · ReLU", fs=12)
sH = 10
box(53.5, yA + hA - sH + 0.4, 16, sH, "Value Stream  $V(s)$", TEAL, sub="64 → 32 → 1", fs=11)
box(53.5, yA - 0.4,          16, sH, "Advantage  $A(s,\\cdot)$", PURPLE, sub="64 → 32 → 8", fs=11)
box(72, yA, 12, hA, "Dueling Combine", NAVY, sub="$Q = V + (A - \\bar{A})$", fs=10.5, sub_fs=10)
box(86.5, yA, 13.5, hA, "Safety Action Mask", ORANGE, sub="util>0.9 → blocked", fs=10.5, sub_fs=9.5)
box(86.5, 86, 13.5, 8.5, "argmax → VM", GREEN, fs=12, sub="real-time dispatch")

arrow(19.4, yA + hA / 2, 21.4, yA + hA / 2)
arrow(33.4, yA + hA / 2, 35.4, yA + hA / 2)
arrow(50.4, yA + hA / 2, 52.9, yA + hA / 2)
arrow(70.1, yA + hA * 0.75, 71.4, yA + hA * 0.75)
arrow(70.1, yA + hA * 0.25, 71.4, yA + hA * 0.25)
arrow(84.4, yA + hA / 2, 85.9, yA + hA / 2)
arrow(93.25, yA + hA + 0.5, 93.25, 85.4, color=GREEN, lw=3.0)

# ================== ROW B: TRAINING LOOP ======================
yB, hB = 40, 13
lane_label(2.6, yB + hB / 2, "TRAINING  LOOP  (300 episodes)")

box(6,  yB, 12, hB, "Environment", BLUE, sub="parallel VMs · Borg tasks", fs=11)
box(21, yB, 14, hB, "Reward", ORANGE, sub="QoS-dominated + shaping", fs=11.5)
box(38, yB, 14, hB, "SumTree PER", TEAL, sub="50k · α=0.6 · β→1", fs=11)
box(55, yB, 15, hB, "Double DQN Target", NAVY, sub="$r+\\gamma\\, Q^-(s',\\, a^*)$", fs=10.5, sub_fs=9.5)
box(73, yB, 12, hB, "Adam Update", NAVY, sub="weighted MSE", fs=11)
box(86.5, yB, 13.5, hB, "Soft Target Sync", PURPLE, sub="$\\tau = 0.005$", fs=11)

arrow(18.4, yB + hB / 2, 20.4, yB + hB / 2)
arrow(35.4, yB + hB / 2, 37.4, yB + hB / 2)
arrow(52.4, yB + hB / 2, 54.4, yB + hB / 2)
arrow(70.4, yB + hB / 2, 72.4, yB + hB / 2)
arrow(85.4, yB + hB / 2, 85.9, yB + hB / 2)

# feedback arrows
arrow(86.5, yB + hB + 0.5, 43, yA - 0.5, color=PURPLE, lw=2.4, ls="--", alpha=0.95)
ax.text(68.5, 55.5, "target-network parameters", fontsize=9, color=PURPLE,
        style="italic", ha="center")
arrow(12, yB + hB + 0.5, 12, yA - 0.5, color=BLUE, lw=2.4, ls=":")
ax.text(13.4, 60.5, "next state $s_{t+1}$", fontsize=9, color=BLUE,
        style="italic", ha="left")

# ================= ROW C: XAI + TRUST =========================
yC, hC = 12, 13
lane_label(2.6, yC + hC / 2, "XAI  +  TRUST  AUDIT")

xai = [("KernelSHAP", "250 coalitions · WLS", BLUE),
       ("Gradient×Input", "exact · 0.15 ms", TEAL),
       ("Occlusion", "windowed ΔQ", GREEN),
       ("Integrated Grads", "path integral", PURPLE)]
bw = 18
for i, (name, sub, c) in enumerate(xai):
    box(6 + i * 19.8, yC, bw, hC, name, c, sub=sub, fs=10.5)
box(86.5, yC, 13.5, hC, "6 Trust Metrics", ORANGE,
    sub="AOPC·fidelity·consistency", fs=10.5, sub_fs=8)

# tap from dueling combine into XAI row
arrow(78, yA - 0.5, 42, yC + hC + 0.5, color=GREY, lw=2.4, ls="--", alpha=0.9)
ax.text(56, 60.5, "explains every dispatch decision  $Q(s, a^*)$",
        fontsize=9.5, color=GREY, style="italic", ha="center")
for i in range(3):
    arrow(24.2 + i * 19.8, yC + hC / 2, 25.4 + i * 19.8, yC + hC / 2, color=GREY, lw=2.0, ls=":")
arrow(79.6, yC + hC / 2, 85.9, yC + hC / 2, color=GREY, lw=2.0, ls=":")

# ============================ FOOTER ===========================
ax.text(54, 5.8, "Ablation-validated: removing Dueling streams or the safety mask each costs ~35% "
        "higher cost  (5 seeds, p < 0.001)",
        ha="center", fontsize=10.5, color=NAVY, style="italic",
        bbox=dict(boxstyle="round,pad=0.55", fc=LIGHT, ec=BLUE, lw=1.2))

handles = [
    Line2D([0], [0], color=BLUE,   lw=8, label="Data / environment"),
    Line2D([0], [0], color=NAVY,   lw=8, label="Network / training"),
    Line2D([0], [0], color=TEAL,   lw=8, label="Value / replay"),
    Line2D([0], [0], color=PURPLE, lw=8, label="Advantage / target net"),
    Line2D([0], [0], color=ORANGE, lw=8, label="Mask / reward / audit"),
    Line2D([0], [0], color=GREEN,  lw=8, label="Decision"),
]
ax.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
          fontsize=10, bbox_to_anchor=(0.5, 0.015))

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved -> {OUT}")
