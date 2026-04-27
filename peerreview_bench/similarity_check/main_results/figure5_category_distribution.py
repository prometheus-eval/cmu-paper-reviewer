#!/usr/bin/env python3
"""
Generate figure5.{png,svg} -- stacked bar chart showing the 4-way
review-item overlap distribution per pair type.

Three bars (inter-reviewer only, no self-similarity):
  - H-H (different reviewers)
  - A-A (different models)
  - H-A

Categories use descriptive overlap labels.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "outputs", "full_similarity", "pairs_llm_azure_ai__gpt-5_4.jsonl",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "figure5.png")
OUT_SVG = os.path.join(OUT_DIR, "figure5.svg")

# Papers dropped due to license restrictions (not confirmed CC BY 4.0)
DROPPED_PAPER_IDS = {11, 20, 22}

# 4-way labels (bottom to top in stacked bar)
NEAR_PARAPHRASE = "same subject, same argument, same evidence"
CONVERGENT = "same subject, same argument, different evidence"
TOPICAL = "same subject, different argument"
UNRELATED = "different subject"

CATEGORY_ORDER = [NEAR_PARAPHRASE, CONVERGENT, TOPICAL, UNRELATED]
CATEGORY_DISPLAY = {
    NEAR_PARAPHRASE: "Same issue, same criticism,\nsame evidence",
    CONVERGENT: "Same issue, same criticism,\ndifferent evidence",
    TOPICAL: "Same issue,\ndifferent criticism",
    UNRELATED: "Different issue",
}

# Short labels for in-bar annotations
CATEGORY_SHORT = {
    NEAR_PARAPHRASE: "Same crit.\nsame evid.",
    CONVERGENT: "Same crit.\ndiff. evid.",
    TOPICAL: "Diff.\ncrit.",
    UNRELATED: "Different\nissue",
}

# Pastel gradient: darkest (most overlap) to lightest (least overlap)
COLORS = {
    NEAR_PARAPHRASE: "#2166ac",  # dark blue
    CONVERGENT: "#67a9cf",       # medium blue
    TOPICAL: "#d1e5f0",          # light blue
    UNRELATED: "#f0f0f0",        # very light grey
}


# ── load ───────────────────────────────────────────────────────────────
def load_pairs(path):
    pairs = []
    n_err = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") is not None and str(r["error"]) != "None":
                n_err += 1
                continue
            if r.get("paper_id") in DROPPED_PAPER_IDS:
                continue
            pairs.append(r)
    print(f"Loaded {len(pairs)} valid pairs, skipped {n_err} errors.")
    return pairs


# ── grouping (inter-reviewer only) ────────────────────────────────────
BAR_ORDER = [
    ("Human–Human", lambda p: p["pair_type"] == "H-H" and not p["same_reviewer"]),
    ("AI–AI", lambda p: p["pair_type"] == "A-A" and not p["same_reviewer"]),
    ("Human–AI", lambda p: p["pair_type"] == "H-A"),
]


def compute_fractions(pairs):
    """Return dict: bar_label -> {category: fraction}."""
    result = {}
    for label, filt in BAR_ORDER:
        subset = [p for p in pairs if filt(p)]
        n = len(subset)
        if n == 0:
            result[label] = {cat: 0.0 for cat in CATEGORY_ORDER}
            continue
        counts = defaultdict(int)
        for p in subset:
            counts[p["parsed_answer"]] += 1
        fracs = {cat: counts[cat] / n for cat in CATEGORY_ORDER}
        result[label] = fracs
        print(f"{label.replace(chr(10), ' ')}: N={n}")
        for cat in CATEGORY_ORDER:
            print(f"  {CATEGORY_DISPLAY[cat].replace(chr(10), ' '):45s}: {fracs[cat]*100:5.1f}%")
        similar = fracs[NEAR_PARAPHRASE] + fracs[CONVERGENT]
        print(f"  {'Overlapping (similar)':45s}: {similar*100:5.1f}%")
    return result


# ── plot ───────────────────────────────────────────────────────────────
def make_figure(fractions):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    bar_labels = [label for label, _ in BAR_ORDER]
    x = np.arange(len(bar_labels))
    width = 0.55

    bottoms = np.zeros(len(bar_labels))

    for cat in CATEGORY_ORDER:
        vals = np.array([fractions[label][cat] for label in bar_labels])
        ax.bar(x, vals, width, bottom=bottoms, color=COLORS[cat],
               edgecolor="white", linewidth=0.3,
               label=CATEGORY_DISPLAY[cat])

        # Percentage labels on each segment (only if >= 2.5%)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.025:
                pct_str = f"{v*100:.1f}%"
                y_center = b + v / 2
                fontsize = 7 if v >= 0.05 else 6
                # Dark text on light segments, white text on dark segments
                text_color = "white" if cat in (NEAR_PARAPHRASE, CONVERGENT) else "#333333"
                ax.text(x[i], y_center, pct_str, ha="center", va="center",
                        fontsize=fontsize, color=text_color, fontweight="bold")

        bottoms += vals

    ax.set_ylabel("Fraction of pairs", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Legend above the plot with enough space
    handles = [Patch(facecolor=COLORS[cat], edgecolor="white", linewidth=0.3,
                     label=CATEGORY_DISPLAY[cat]) for cat in CATEGORY_ORDER]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.08),
              ncol=2, fontsize=7.5, frameon=False, columnspacing=1.5,
              handletextpad=0.4, handlelength=1.2, labelspacing=1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(top=0.72)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


# ── main ───────────────────────────────────────────────────────────────
def main():
    pairs = load_pairs(DATA_FILE)
    fractions = compute_fractions(pairs)
    make_figure(fractions)


if __name__ == "__main__":
    main()
