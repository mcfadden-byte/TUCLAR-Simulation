"""Three result charts against link availability: mission time, constraint violations, communication volume.

  python plot_results.py results/results.csv results/figs
Needs matplotlib. Communication volume = messages received per mission x message size:
22 bits for the task-tree message (HTN-only, Ours); 660 bits for LLM-only's natural-language message, the mean
measured over 2208 messages written by gemma in results/llm_nl (method llm_nl, 177 runs).
"""
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sar_sim as S  # noqa: E402

NL_BITS = 660
QS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# method key, label, colour (reference categorical slots 1-3), marker
METHODS = [("node", "Ours", "#2a78d6", "o"), ("llm", "LLM-only", "#eb6834", "s"), ("htn", "HTN-only", "#1baf7a", "^")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "savefig.dpi": 200, "figure.facecolor": "white"})


def load(path):
    runs = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            runs[(r["method"], float(r["q"]))][int(r["seed"])] = r
    return runs


def messages_received(seed, q, rounds, decisions):
    """Per-round state messages over the links that are up (exact, from the link model) plus the update each
    decision sends to reachable teammates (expected value, 4 teammates x q)."""
    links, n = S.Links(seed, q), 0
    for _ in range(rounds):
        n += 2 * sum(links.up.values())
        links.step()
    return n + decisions * 4 * q


def axes(title, ylabel):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.set_title(title, pad=12, color=INK)
    ax.set_xlabel("Link availability (fraction of time a link is up)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(QS)
    ax.set_xticklabels(["0\n(no links)", "0.2", "0.4", "0.6", "0.8", "1\n(always up)"])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def note(ax, text):
    ax.text(0, -0.30, text, transform=ax.transAxes, fontsize=9, color=INK2, va="top")


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=10,
                fontweight="bold" if text.startswith("Ours") else "normal")


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main(csv_path, out):
    os.makedirs(out, exist_ok=True)
    runs = load(csv_path)

    # 1. Mission time, mean with 95% CI
    fig, ax = axes("Time to find the missing person", "Rounds until found (log scale)")
    for key, label, color, marker in METHODS:
        mean, lo, hi = [], [], []
        for q in QS:
            v = [float(r["rounds"]) for r in runs[(key, q)].values()]
            m, h = st.mean(v), 1.96 * st.stdev(v) / math.sqrt(len(v))
            mean.append(m), lo.append(m - h), hi.append(m + h)
        ax.fill_between(QS, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(QS, mean, color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor="white",
                markeredgewidth=1.2, label=label, zorder=3)
        end_label(ax, QS[-1], mean[-1] * {"Ours": 0.93, "LLM-only": 1.04, "HTN-only": 1.07}[label],
                  f"{label} {mean[-1]:.1f}", color)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([15, 20, 30, 50, 100, 150]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["15", "20", "30", "50", "100", "150"])
    ax.set_ylim(15, 160)
    ax.set_xlim(-0.05, 1.27)
    ax.legend(loc="upper right")
    note(ax, "Mean over 100 missions per point; shaded band = 95% confidence interval.")
    save(fig, out, "fig1_mission_time")

    # 2. Constraint violations: drones that ran out of battery away from the base
    fig, ax = axes("Constraint violations", "Crashed drones per 100 missions")
    w = 0.055
    for k, (key, label, color, marker) in enumerate(METHODS):
        vals = [sum(float(r["crashes"]) for r in runs[(key, q)].values()) for q in QS]
        xs = [q + (k - 1) * w for q in QS]
        ax.bar(xs, vals, width=w - 0.008, color=color, label=f"{label} (total {sum(vals):.0f})", zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.06, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlim(-0.12, 1.12)
    ax.legend(loc="upper right")
    note(ax, "A crash: a drone's battery runs out away from the base. 100 missions of 5 drones per bar.")
    save(fig, out, "fig2_constraint_violations")

    # 3. Communication volume per mission
    fig, ax = axes("Communication per mission", "Data received by teammates (kbit)")
    final = {}
    for key, label, color, marker in METHODS:
        bits = NL_BITS if key == "llm" else S.STATE_BITS
        vals = []
        for q in QS:
            rs = runs[(key, q)]
            msgs = st.mean(messages_received(s, q, int(r["rounds"]), float(r["decisions"])) for s, r in rs.items())
            vals.append(msgs * bits / 1000)
        final[label] = vals[-1]
        ax.plot(QS, vals, color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor="white",
                markeredgewidth=1.2, label=label, zorder=4 if key == "node" else 3)
    ax.annotate(f"LLM-only {final['LLM-only']:.0f}", (1.0, final["LLM-only"]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=10, color=INK)
    ax.annotate(f"HTN-only {final['HTN-only']:.1f}", (1.0, final["HTN-only"]), xytext=(8, 17),
                textcoords="offset points", va="center", fontsize=10, color=INK)
    ax.annotate(f"Ours {final['Ours']:.1f}", (1.0, final["Ours"]), xytext=(8, 3),
                textcoords="offset points", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlim(-0.05, 1.27)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    note(ax, f"Message size: Ours and HTN-only send a {S.STATE_BITS}-bit task message; LLM-only's natural-language\n"
             f"messages average {NL_BITS} bits (13 words). Same link model and message schedule for all.")
    save(fig, out, "fig3_communication")
    print("saved to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
