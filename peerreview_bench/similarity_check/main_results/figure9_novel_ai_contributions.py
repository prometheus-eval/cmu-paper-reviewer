#!/usr/bin/env python3
"""
Generate figure9.pdf -- grouped bar chart showing quality of AI review items
split by whether they are "matched" (similar to at least one human item) or
"uncovered" (no human match), compared against the human baseline.

For each group (matched AI, uncovered AI, human baseline):
  - "Correct" rate
  - "Fully good" rate (Correct + Significant + Sufficient)
  - "Significant | Correct" rate (among correct items, fraction that are Significant)

Paper-level bootstrap (10,000 iterations) for 95% CIs.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "outputs", "full_similarity", "pairs_llm_azure_ai__gpt-5_4.jsonl",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(OUT_DIR, "figure9.pdf")

N_BOOT = 10_000
SEED = 42

# 4-way ordinal ranking (higher = more similar)
ORDINAL = {
    "different subject": 0,
    "same subject, different argument": 1,
    "same subject, same argument, different evidence": 2,
    "same subject, same argument, same evidence": 3,
}

# Papers dropped due to license restrictions (not confirmed CC BY 4.0)
DROPPED_PAPER_IDS = {11, 20, 22}


# ── load similarity pairs ────────────────────────────────────────────
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


# ── load expert annotations (primary only) ───────────────────────────
def load_annotations():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    from load_data import load_expert_annotation_rows
    rows = load_expert_annotation_rows()
    primary = [r for r in rows if r["annotator_source"] == "primary"]
    return primary


def is_fully_good(row):
    return (row.get("correctness") == "Correct"
            and row.get("significance") == "Significant"
            and row.get("evidence") == "Sufficient")


# ── classify AI items as matched/uncovered ──────────────────────────────
def classify_ai_items(pairs, annotations):
    """Classify each AI item as matched or uncovered based on H-A similarity.

    Returns:
        matched_ai: list of (paper_id, reviewer_id, item_number) for matched AI items
        uncovered_ai: list of (paper_id, reviewer_id, item_number) for uncovered AI items
        annot_lookup: {(paper_id, reviewer_id, item_number): annotation_row}
    """
    # Build annotation lookup
    annot_lookup = {}
    for r in annotations:
        key = (r["paper_id"], r["reviewer_id"], r["review_item_number"])
        annot_lookup[key] = r

    # For each AI item, find max similarity to any human item (via H-A pairs)
    # In H-A pairs: item_a is Human, item_b is AI
    ai_max_sim = defaultdict(int)  # (paper_id, ai_rev, ai_item) -> max ordinal
    ai_items_seen = set()

    for p in pairs:
        if p["pair_type"] != "H-A":
            continue
        pid = p["paper_id"]
        ai_rev = p["item_b"]["reviewer_id"]
        ai_item = p["item_b"]["review_item_number"]
        key = (pid, ai_rev, ai_item)
        ai_items_seen.add(key)

        ordinal = ORDINAL.get(p["parsed_answer"], 0)
        ai_max_sim[key] = max(ai_max_sim[key], ordinal)

    # Split into matched (max ordinal >= 2, i.e., "similar") and uncovered
    matched_ai = []
    uncovered_ai = []
    for key in sorted(ai_items_seen):
        max_ord = ai_max_sim.get(key, 0)
        # "similar" means ordinal >= 2 (convergent or near-paraphrase)
        if max_ord >= 2:
            matched_ai.append(key)
        else:
            uncovered_ai.append(key)

    # Also collect all human items
    human_items = set()
    for p in pairs:
        if p["pair_type"] == "H-A":
            pid = p["paper_id"]
            h_rev = p["item_a"]["reviewer_id"]
            h_item = p["item_a"]["review_item_number"]
            human_items.add((pid, h_rev, h_item))

    human_items = sorted(human_items)

    return matched_ai, uncovered_ai, human_items, annot_lookup


# ── compute quality metrics for a group ──────────────────────────────
def compute_quality(items, annot_lookup):
    """Compute correct rate, fully-good rate, significant|correct rate."""
    n_total = 0
    n_correct = 0
    n_fully_good = 0
    n_significant_given_correct = 0
    n_correct_for_cond = 0

    for key in items:
        row = annot_lookup.get(key)
        if row is None:
            continue
        n_total += 1
        if row["correctness"] == "Correct":
            n_correct += 1
            n_correct_for_cond += 1
            if row["significance"] == "Significant":
                n_significant_given_correct += 1
        if is_fully_good(row):
            n_fully_good += 1

    if n_total == 0:
        return {"correct": 0.0, "fully_good": 0.0, "sig_given_correct": 0.0, "n": 0}

    return {
        "correct": n_correct / n_total,
        "fully_good": n_fully_good / n_total,
        "sig_given_correct": (n_significant_given_correct / n_correct_for_cond
                              if n_correct_for_cond > 0 else 0.0),
        "n": n_total,
    }


# ── paper-level bootstrap ───────────────────────────────────────────
def bootstrap_quality(matched_ai, uncovered_ai, human_items, annot_lookup,
                      n_boot=N_BOOT, seed=SEED):
    """Paper-level bootstrap for quality metrics across 3 groups."""
    # Group items by paper
    def by_paper(items):
        d = defaultdict(list)
        for key in items:
            d[key[0]].append(key)
        return d

    matched_by_paper = by_paper(matched_ai)
    uncovered_by_paper = by_paper(uncovered_ai)
    human_by_paper = by_paper(human_items)

    # All paper IDs across all groups
    all_pids = sorted(set(matched_by_paper.keys())
                      | set(uncovered_by_paper.keys())
                      | set(human_by_paper.keys()))
    n_papers = len(all_pids)
    paper_arr = np.array(all_pids)

    # Observed
    obs_matched = compute_quality(matched_ai, annot_lookup)
    obs_uncovered = compute_quality(uncovered_ai, annot_lookup)
    obs_human = compute_quality(human_items, annot_lookup)

    metrics = ["correct", "fully_good", "sig_given_correct"]
    groups = ["matched", "uncovered", "human"]
    boot = {f"{g}_{m}": np.empty(n_boot) for g in groups for m in metrics}

    rng = np.random.RandomState(seed)

    for b in range(n_boot):
        sampled = rng.choice(paper_arr, size=n_papers, replace=True)

        b_matched = []
        b_uncovered = []
        b_human = []
        for pid in sampled:
            b_matched.extend(matched_by_paper.get(pid, []))
            b_uncovered.extend(uncovered_by_paper.get(pid, []))
            b_human.extend(human_by_paper.get(pid, []))

        q_m = compute_quality(b_matched, annot_lookup)
        q_o = compute_quality(b_uncovered, annot_lookup)
        q_h = compute_quality(b_human, annot_lookup)

        for m in metrics:
            boot[f"matched_{m}"][b] = q_m[m]
            boot[f"uncovered_{m}"][b] = q_o[m]
            boot[f"human_{m}"][b] = q_h[m]

    obs = {"matched": obs_matched, "uncovered": obs_uncovered, "human": obs_human}
    cis = {}
    for key, arr in boot.items():
        cis[key] = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    return obs, cis


# ── plot ─────────────────────────────────────────────────────────────
def make_figure(obs, cis):
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

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    groups = ["matched", "uncovered", "human"]
    group_labels = ["Matched AI items", "Uncovered AI items", "Human items\n(baseline)"]
    metrics = ["correct", "fully_good", "sig_given_correct"]
    metric_labels = ["Correct", "Fully good", "Significant | Correct"]
    colors = ["#0072B2", "#E69F00", "#009E73"]

    x = np.arange(len(groups))
    n_bars = len(metrics)
    width = 0.22

    for i, (metric, m_label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [obs[g][metric] * 100 for g in groups]
        errs_lo = [obs[g][metric] * 100 - cis[f"{g}_{metric}"][0] * 100 for g in groups]
        errs_hi = [cis[f"{g}_{metric}"][1] * 100 - obs[g][metric] * 100 for g in groups]

        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      yerr=[errs_lo, errs_hi],
                      capsize=3, color=color, edgecolor="white", linewidth=0.3,
                      label=m_label, error_kw={"linewidth": 0.8})

        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.2,
                    f"{height:.1f}%", ha="center", va="bottom", fontsize=6.5)

    ax.set_ylabel("Rate (%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.15), ncol=3, columnspacing=1.2,
              handletextpad=0.4, handlelength=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nWrote {OUT_PDF}")


# ── main ─────────────────────────────────────────────────────────────
def main():
    pairs = load_pairs(DATA_FILE)
    annotations = load_annotations()

    matched_ai, uncovered_ai, human_items, annot_lookup = classify_ai_items(pairs, annotations)

    print(f"\nMatched AI items: {len(matched_ai)}")
    print(f"Uncovered AI items:  {len(uncovered_ai)}")
    print(f"Human items:      {len(human_items)}")
    print(f"Annotation lookup size: {len(annot_lookup)}")

    obs, cis = bootstrap_quality(matched_ai, uncovered_ai, human_items, annot_lookup)

    print("\n=== Quality of AI Items by Match Status ===")
    for group, label in [("matched", "Matched AI"), ("uncovered", "Uncovered AI"),
                         ("human", "Human (baseline)")]:
        print(f"\n  {label} (n={obs[group]['n']}):")
        for metric, m_label in [("correct", "Correct"),
                                ("fully_good", "Fully good"),
                                ("sig_given_correct", "Significant | Correct")]:
            ci = cis[f"{group}_{metric}"]
            print(f"    {m_label}: {obs[group][metric]*100:.1f}%  "
                  f"95% CI [{ci[0]*100:.1f}, {ci[1]*100:.1f}]")

    make_figure(obs, cis)


if __name__ == "__main__":
    main()
