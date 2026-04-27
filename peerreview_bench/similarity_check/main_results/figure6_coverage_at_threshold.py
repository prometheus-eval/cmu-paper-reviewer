#!/usr/bin/env python3
"""
Generate figure6.{png,svg} -- coverage bar chart with scaling effect.

Four groups showing how coverage changes with 1 vs 3 reviewers:
  1. 1 human → 1 AI: avg coverage across all 9 (Human_i, AI_j) pairs
  2. 3 humans → 3 AIs: pool all human items, check any AI covers each
  3. 1 AI → 1 human: avg coverage across all 9 (AI_j, Human_i) pairs
  4. 3 AIs → 3 humans: pool all AI items, check any human covers each

Each group has 3 bars at different overlap levels.
"""

import json
import os
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
OUT_PNG = os.path.join(OUT_DIR, "figure6.png")
OUT_SVG = os.path.join(OUT_DIR, "figure6.svg")

# Papers dropped due to license restrictions (not confirmed CC BY 4.0)
DROPPED_PAPER_IDS = {11, 20, 22}

N_BOOT = 10_000
SEED = 42

ORDINAL = {
    "same subject, same argument, same evidence": 3,
    "same subject, same argument, different evidence": 2,
    "same subject, different argument": 1,
    "different subject": 0,
}

THRESHOLD_LABELS = {
    1: "Same issue",
    2: "Same issue,\nsame criticism",
    3: "Same issue,\nsame criticism,\nsame evidence",
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


# ── build per-pair similarity lookup ──────────────────────────────────
def build_similarity_index(pairs):
    """Build: (paper_id, source_reviewer, source_item, target_reviewer) -> max ordinal.
    Includes both H-A and H-H pairs."""
    sim = defaultdict(lambda: 0)

    for p in pairs:
        if p["pair_type"] not in ("H-A", "H-H"):
            continue
        # For H-H, skip same-reviewer pairs
        if p["pair_type"] == "H-H" and p["same_reviewer"]:
            continue
        ov = ORDINAL[p["parsed_answer"]]

        a = p["item_a"]
        b = p["item_b"]

        key_ab = (p["paper_id"], a["reviewer_id"], a["review_item_number"], b["reviewer_id"])
        sim[key_ab] = max(sim[key_ab], ov)

        key_ba = (p["paper_id"], b["reviewer_id"], b["review_item_number"], a["reviewer_id"])
        sim[key_ba] = max(sim[key_ba], ov)

    return dict(sim)


def get_items_by_paper(pairs):
    """Return per-paper sets of (reviewer_id, item_number, reviewer_type)."""
    items_by_paper = defaultdict(set)
    for p in pairs:
        if p["pair_type"] not in ("H-A", "H-H"):
            continue
        for side in ("item_a", "item_b"):
            it = p[side]
            items_by_paper[p["paper_id"]].add(
                (it["reviewer_id"], it["review_item_number"], it["reviewer_type"])
            )
    return dict(items_by_paper)


# ── coverage computations ─────────────────────────────────────────────
def coverage_1to1(sim_index, items_by_paper, source_type, target_type, threshold):
    """Average coverage across all (source_reviewer, target_reviewer) pairs.
    For each pair, compute fraction of source items covered by that target."""
    all_pair_coverages = []  # list of (paper_id, coverage) for bootstrap

    by_paper = defaultdict(list)

    for pid, items in items_by_paper.items():
        source_reviewers = set(rid for rid, _, rtype in items if rtype == source_type)
        target_reviewers = set(rid for rid, _, rtype in items if rtype == target_type)

        for srev in source_reviewers:
            source_items = [(rid, inum) for rid, inum, rtype in items
                           if rtype == source_type and rid == srev]
            for trev in target_reviewers:
                covered = 0
                total = len(source_items)
                for srid, sinum in source_items:
                    max_sim = sim_index.get((pid, srid, sinum, trev), 0)
                    if max_sim >= threshold:
                        covered += 1
                if total > 0:
                    by_paper[pid].append(covered / total)

    # Overall average
    all_vals = []
    for pid, coverages in by_paper.items():
        all_vals.extend(coverages)
    obs = np.mean(all_vals) if all_vals else 0.0

    return obs, dict(by_paper)


def coverage_poolsrc_1tgt(sim_index, items_by_paper, source_type, target_type, threshold):
    """Pool all source items, check coverage by each single target reviewer,
    then average across target reviewers."""
    by_paper = defaultdict(list)

    for pid, items in items_by_paper.items():
        source_items = [(rid, inum) for rid, inum, rtype in items if rtype == source_type]
        target_reviewers = set(rid for rid, _, rtype in items if rtype == target_type)

        for trev in target_reviewers:
            covered = 0
            total = len(source_items)
            for srid, sinum in source_items:
                max_sim = sim_index.get((pid, srid, sinum, trev), 0)
                if max_sim >= threshold:
                    covered += 1
            if total > 0:
                by_paper[pid].append(covered / total)

    all_vals = []
    for vals in by_paper.values():
        all_vals.extend(vals)
    obs = np.mean(all_vals) if all_vals else 0.0

    return obs, dict(by_paper)


def coverage_all_to_all(sim_index, items_by_paper, source_type, target_type, threshold):
    """For each source item, check if ANY target reviewer covers it."""
    by_paper = defaultdict(list)

    for pid, items in items_by_paper.items():
        source_items = [(rid, inum) for rid, inum, rtype in items if rtype == source_type]
        target_reviewers = set(rid for rid, _, rtype in items if rtype == target_type)

        for srid, sinum in source_items:
            max_sim = 0
            for trev in target_reviewers:
                s = sim_index.get((pid, srid, sinum, trev), 0)
                max_sim = max(max_sim, s)
            by_paper[pid].append(1 if max_sim >= threshold else 0)

    # Overall
    all_vals = []
    for vals in by_paper.values():
        all_vals.extend(vals)
    obs = np.mean(all_vals) if all_vals else 0.0

    return obs, dict(by_paper)


def paper_bootstrap(by_paper, n_boot=N_BOOT, seed=SEED):
    """Paper-level bootstrap CI."""
    paper_ids = sorted(by_paper.keys())
    n_papers = len(paper_ids)
    rng = np.random.RandomState(seed)

    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(paper_ids, size=n_papers, replace=True)
        vals = []
        for pid in sampled:
            vals.extend(by_paper[pid])
        boot[b] = np.mean(vals) if vals else 0.0

    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ── plot ───────────────────────────────────────────────────────────────
def make_figure(pairs):
    sim_index = build_similarity_index(pairs)
    items_by_paper = get_items_by_paper(pairs)

    thresholds = [1, 2, 3]

    groups = [
        ("1 human reviewer\ncovered by\n1 human reviewer", "1to1_hh", "Human", "Human"),
        ("1 human reviewer\ncovered by\n1 AI model", "1to1", "Human", "AI"),
        ("3 human reviewers\ncovered by\n3 AI models", "all", "Human", "AI"),
    ]

    data = {}  # (group_label, threshold) -> (obs, lo, hi)
    for g_label, mode, src, tgt in groups:
        for t in thresholds:
            if mode == "1to1" or mode == "1to1_hh":
                obs, bp = coverage_1to1(sim_index, items_by_paper, src, tgt, t)
            elif mode == "pool_src_1tgt":
                obs, bp = coverage_poolsrc_1tgt(sim_index, items_by_paper, src, tgt, t)
            else:
                obs, bp = coverage_all_to_all(sim_index, items_by_paper, src, tgt, t)
            lo, hi = paper_bootstrap(bp)
            data[(g_label, t)] = (obs, lo, hi)
            print(f"  {g_label.replace(chr(10), ' ')} @ {THRESHOLD_LABELS[t].replace(chr(10), ' ')}: "
                  f"{obs*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]")

    # Plot
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

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    group_labels = [gl for gl, _, _, _ in groups]
    n_groups = len(group_labels)
    bar_width = 0.2
    group_positions = np.arange(n_groups) * 1.2

    bar_colors = ["#d1e5f0", "#67a9cf", "#2166ac"]

    for i, t in enumerate(thresholds):
        offsets = group_positions + (i - 1) * bar_width
        vals = [data[(gl, t)][0] for gl in group_labels]
        lo = [data[(gl, t)][0] - data[(gl, t)][1] for gl in group_labels]
        hi = [data[(gl, t)][2] - data[(gl, t)][0] for gl in group_labels]

        ax.bar(offsets, vals, bar_width, color=bar_colors[i],
               edgecolor="white", linewidth=0.3,
               yerr=[lo, hi], capsize=3, error_kw={"linewidth": 0.8},
               label=THRESHOLD_LABELS[t])

        for j, v in enumerate(vals):
            ax.text(offsets[j], v + hi[j] + 0.02, f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Coverage (fraction of items)", fontsize=10)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax.legend(loc="upper left", fontsize=8, frameon=True,
              fancybox=False, edgecolor="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


# ── main ───────────────────────────────────────────────────────────────
def main():
    pairs = load_pairs(DATA_FILE)
    make_figure(pairs)


if __name__ == "__main__":
    main()
