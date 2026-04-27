#!/usr/bin/env python3
"""
Generate figure7.pdf -- similarity x correctness analysis.

For each AI review item:
  1. Find max similarity to any Human item on the same paper (4-way ordinal).
  2. Split into "has human match" (max >= 2) vs "no human match" (max <= 1).
  3. Look up correctness from expert_annotation (annotator_source='primary').
  4. Plot Correct rate for each group with paper-level bootstrap CIs.

Symmetric analysis: for each Human item, max similarity to any AI item.

Uses load_expert_annotation_rows() from load_data.py for correctness labels.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "outputs", "full_similarity", "pairs_llm_azure_ai__gpt-5_4.jsonl",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(OUT_DIR, "figure7.pdf")

# Papers dropped due to license restrictions (not confirmed CC BY 4.0)
DROPPED_PAPER_IDS = {11, 20, 22}

REPO_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.insert(0, os.path.abspath(REPO_ROOT))

N_BOOT = 10_000
SEED = 42

# Ordinal mapping
ORDINAL = {
    "same subject, same argument, same evidence": 3,
    "same subject, same argument, different evidence": 2,
    "same subject, different argument": 1,
    "different subject": 0,
}

MATCH_THRESHOLD = 2  # convergent or near-paraphrase


# ── load similarity pairs ─────────────────────────────────────────────
def load_pairs(path):
    pairs = []
    n_err = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["error"] is not None:
                n_err += 1
                continue
            if r.get("paper_id") in DROPPED_PAPER_IDS:
                continue
            pairs.append(r)
    print(f"Loaded {len(pairs)} valid pairs, skipped {n_err} errors.")
    return pairs


# ── load expert annotation ────────────────────────────────────────────
def load_correctness():
    """Load correctness labels from expert_annotation (primary only).

    Returns: dict (paper_id, reviewer_id, review_item_number) -> correctness string.
    """
    from load_data import load_expert_annotation_rows
    rows = load_expert_annotation_rows()
    corr = {}
    for r in rows:
        if r["annotator_source"] != "primary":
            continue
        key = (int(r["paper_id"]), r["reviewer_id"], int(r["review_item_number"]))
        corr[key] = r["correctness"]
    print(f"Loaded {len(corr)} primary correctness labels.")
    return corr


# ── max similarity computation ────────────────────────────────────────
def compute_max_sim(pairs, source_type, target_type):
    """For each (paper, source-item), compute max ordinal similarity to any
    target-item in the same paper. Uses H-A pairs only.

    Returns: dict (paper_id, reviewer_id, item_number) -> max ordinal.
    """
    max_sim = {}

    for p in pairs:
        if p["pair_type"] != "H-A":
            continue
        ord_val = ORDINAL[p["parsed_answer"]]

        a_type = p["item_a"]["reviewer_type"]
        b_type = p["item_b"]["reviewer_type"]

        if a_type == source_type and b_type == target_type:
            key = (p["paper_id"], p["item_a"]["reviewer_id"],
                   p["item_a"]["review_item_number"])
            max_sim[key] = max(max_sim.get(key, 0), ord_val)

        if b_type == source_type and a_type == target_type:
            key = (p["paper_id"], p["item_b"]["reviewer_id"],
                   p["item_b"]["review_item_number"])
            max_sim[key] = max(max_sim.get(key, 0), ord_val)

    return max_sim


# ── join similarity with correctness ──────────────────────────────────
def join_sim_correctness(max_sim, corr_labels, source_type):
    """Join similarity with correctness. Returns by_paper dict:
    paper_id -> list of (has_match: bool, is_correct: bool).
    """
    by_paper = defaultdict(list)
    matched = 0
    unmatched = 0
    missing_corr = 0

    for (pid, rid, inum), sim_val in max_sim.items():
        key = (pid, rid, inum)
        if key not in corr_labels:
            missing_corr += 1
            continue
        corr = corr_labels[key]
        is_correct = corr == "Correct"
        has_match = sim_val >= MATCH_THRESHOLD
        by_paper[pid].append((has_match, is_correct))
        if has_match:
            matched += 1
        else:
            unmatched += 1

    print(f"  {source_type} items: {matched} matched, {unmatched} unmatched, "
          f"{missing_corr} missing correctness label")
    return dict(by_paper)


# ── paper-level bootstrap for correctness rate ────────────────────────
def paper_bootstrap_correct_rate(by_paper, has_match_filter, n_boot=N_BOOT, seed=SEED):
    """Paper-level bootstrap for the correctness rate of items with
    has_match == has_match_filter.

    Returns: (obs, ci_lo, ci_hi, n_items).
    """
    paper_ids = sorted(by_paper.keys())
    n_papers = len(paper_ids)
    rng = np.random.RandomState(seed)

    # Observed
    total = 0
    correct = 0
    for vals in by_paper.values():
        for hm, ic in vals:
            if hm == has_match_filter:
                total += 1
                if ic:
                    correct += 1
    obs = correct / total if total > 0 else 0.0

    # Bootstrap
    paper_id_arr = np.array(paper_ids)
    boot = np.empty(n_boot)

    for b in range(n_boot):
        sampled = rng.choice(paper_id_arr, size=n_papers, replace=True)
        b_total = 0
        b_correct = 0
        for pid in sampled:
            if pid not in by_paper:
                continue
            for hm, ic in by_paper[pid]:
                if hm == has_match_filter:
                    b_total += 1
                    if ic:
                        b_correct += 1
        boot[b] = b_correct / b_total if b_total > 0 else 0.0

    ci_lo = np.percentile(boot, 2.5)
    ci_hi = np.percentile(boot, 97.5)
    return obs, ci_lo, ci_hi, total


# ── plot ───────────────────────────────────────────────────────────────
def make_figure(pairs, corr_labels):
    # Compute max similarities
    ai_max_sim = compute_max_sim(pairs, "AI", "Human")
    human_max_sim = compute_max_sim(pairs, "Human", "AI")

    print("\nAI items (similarity to Human):")
    ai_by_paper = join_sim_correctness(ai_max_sim, corr_labels, "AI")

    print("Human items (similarity to AI):")
    human_by_paper = join_sim_correctness(human_max_sim, corr_labels, "Human")

    # Compute correctness rates
    results = {}  # (source, match_status) -> (obs, lo, hi, n)
    for label, bp, source in [
        ("AI items", ai_by_paper, "AI"),
        ("Human items", human_by_paper, "Human"),
    ]:
        for match_status, match_filter in [
            ("Has human match", True) if source == "AI" else ("Has AI match", True),
            ("No human match", False) if source == "AI" else ("No AI match", False),
        ]:
            obs, lo, hi, n = paper_bootstrap_correct_rate(bp, match_filter)
            results[(label, match_status)] = (obs, lo, hi, n)
            print(f"  {label} / {match_status}: "
                  f"Correct={obs*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}] (N={n})")

    # Publication style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    # 2 groups (AI items, Human items), each with 2 bars (matched / unmatched)
    group_labels = ["AI items", "Human items"]
    match_labels_ai = ["Has human match", "No human match"]
    match_labels_human = ["Has AI match", "No AI match"]

    n_groups = 2
    bar_width = 0.3
    group_positions = np.arange(n_groups)

    # Colors
    color_matched = "#0072B2"   # blue
    color_unmatched = "#D55E00"  # vermilion

    for i, (match_status_idx, color, bar_label) in enumerate([
        (0, color_matched, "Matched (sim $\\geq$ 2)"),
        (1, color_unmatched, "Unmatched (sim $\\leq$ 1)"),
    ]):
        offsets = group_positions + (i - 0.5) * bar_width

        vals = []
        errs_lo = []
        errs_hi = []
        n_items = []

        for g, gl in enumerate(group_labels):
            if gl == "AI items":
                ms = match_labels_ai[match_status_idx]
            else:
                ms = match_labels_human[match_status_idx]
            obs, lo, hi, n = results[(gl, ms)]
            vals.append(obs)
            errs_lo.append(obs - lo)
            errs_hi.append(hi - obs)
            n_items.append(n)

        ax.bar(offsets, vals, bar_width, color=color,
               edgecolor="white", linewidth=0.3,
               yerr=[errs_lo, errs_hi], capsize=3,
               error_kw={"linewidth": 0.8},
               label=bar_label)

        # Value labels
        for j, (v, eh, n) in enumerate(zip(vals, errs_hi, n_items)):
            ax.text(offsets[j], v + eh + 0.02,
                    f"{v*100:.1f}%\n(N={n})",
                    ha="center", va="bottom", fontsize=6.5)

    ax.set_ylabel("Fraction rated Correct", fontsize=9)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
              ncol=2, fontsize=7, frameon=False, columnspacing=1.5,
              handletextpad=0.4, handlelength=1.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nWrote {OUT_PDF}")


# ── main ───────────────────────────────────────────────────────────────
def main():
    pairs = load_pairs(DATA_FILE)
    corr_labels = load_correctness()
    make_figure(pairs, corr_labels)


if __name__ == "__main__":
    main()
