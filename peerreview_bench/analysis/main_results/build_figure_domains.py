#!/usr/bin/env python3
"""
Generate figure_domains.{png,svg} -- donut chart showing the domain
breakdown of the 80 papers in PeerReview Bench across Nature journal
broad subject categories.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "figure_domains.png")
OUT_SVG = os.path.join(OUT_DIR, "figure_domains.svg")

# Domain breakdown (82 papers in the expert annotation study)
# Papers 11, 20, 22 dropped due to license restrictions
DOMAINS = {
    "Physical\nSciences": 38,
    "Biological\nSciences": 30,
    "Health\nSciences": 14,
}

# Nature-inspired colours (subdued, print-friendly)
COLORS = ["#4e79a7", "#59a14f", "#e15759"]


def make_figure():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 300,
    })

    labels = list(DOMAINS.keys())
    sizes = list(DOMAINS.values())
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda pct: f"{int(round(pct * total / 100))}\n({pct:.0f}%)",
        startangle=90,
        colors=COLORS,
        pctdistance=0.72,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )

    # Style the percentage text
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")

    # Add legend-style labels outside
    ax.legend(
        wedges, [f"{l.replace(chr(10), ' ')} ({s})" for l, s in zip(labels, sizes)],
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=9,
        frameon=False,
        labelspacing=1.2,
    )

    # Central text
    ax.text(0, 0, f"{total}\npapers", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333333")

    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


if __name__ == "__main__":
    make_figure()
