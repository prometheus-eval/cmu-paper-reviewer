#!/usr/bin/env python3
"""
Generate figure8.pdf -- grouped bar chart showing what percentage of human
review items are covered by 1 / 2 / 3 AI reviewers, with separate bars
for "all human items" and "fully good human items only".

Also computes the symmetric metric: what fraction of AI items are covered
by at least one human reviewer.

Coverage definition: a human item h is "covered" by AI reviewer R if there
exists at least one H-A pair (h, r) with parsed_binary == "similar" where r
belongs to R.

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
OUT_PDF = os.path.join(OUT_DIR, "figure8.pdf")

N_BOOT = 10_000
SEED = 42

AI_REVIEWERS = ["Claude", "GPT", "Gemini"]

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
    return (row["correctness"] == "Correct"
            and row["significance"] == "Significant"
            and row["evidence"] == "Sufficient")


# ── build coverage structures ────────────────────────────────────────
def build_coverage(pairs, annotations):
    """Return per-paper, per-human-item coverage info.

    Returns:
        human_items_by_paper: {paper_id: [(reviewer_id, item_number), ...]}
        fully_good_set: set of (paper_id, reviewer_id, item_number)
        h_covered_by_ai: {paper_id: {(h_reviewer, h_item): set of AI reviewer_ids}}
        ai_covered_by_h: {paper_id: {(ai_reviewer, ai_item): bool}}
        ai_items_by_paper: {paper_id: [(reviewer_id, item_number), ...]}
    """
    # Build fully-good set from annotations
    fully_good_set = set()
    for r in annotations:
        if is_fully_good(r):
            fully_good_set.add((r["paper_id"], r["reviewer_id"], r["review_item_number"]))

    # Collect all human items and AI items from H-A pairs
    human_items_by_paper = defaultdict(set)
    ai_items_by_paper = defaultdict(set)

    # h_covered_by_ai[paper_id][(h_reviewer, h_item)] = set of AI reviewer_ids that cover it
    h_covered_by_ai = defaultdict(lambda: defaultdict(set))
    # ai_covered_by_h[paper_id][(ai_reviewer, ai_item)] = True if any human covers it
    ai_covered_by_h = defaultdict(lambda: defaultdict(bool))

    for p in pairs:
        if p["pair_type"] != "H-A":
            continue

        pid = p["paper_id"]
        # In H-A pairs, item_a is Human, item_b is AI
        h_rev = p["item_a"]["reviewer_id"]
        h_item = p["item_a"]["review_item_number"]
        ai_rev = p["item_b"]["reviewer_id"]
        ai_item = p["item_b"]["review_item_number"]

        human_items_by_paper[pid].add((h_rev, h_item))
        ai_items_by_paper[pid].add((ai_rev, ai_item))

        if p["parsed_binary"] == "similar":
            h_covered_by_ai[pid][(h_rev, h_item)].add(ai_rev)
            ai_covered_by_h[pid][(ai_rev, ai_item)] = True

    # Convert sets to sorted lists for consistency
    human_items_by_paper = {pid: sorted(items) for pid, items in human_items_by_paper.items()}
    ai_items_by_paper = {pid: sorted(items) for pid, items in ai_items_by_paper.items()}

    return human_items_by_paper, fully_good_set, h_covered_by_ai, ai_covered_by_h, ai_items_by_paper


# ── compute per-paper coverage stats ────────────────────────────────
def compute_paper_stats(paper_ids, human_items_by_paper, fully_good_set,
                        h_covered_by_ai, ai_covered_by_h, ai_items_by_paper):
    """For the given set of paper_ids, compute coverage fractions.

    Returns dict of metric_name -> value.
    """
    # ── Human items covered by k AI reviewers ──
    # For "1 AI reviewer": for each human item, check each AI reviewer
    #   individually; average across all (human_item, ai_reviewer) combos.
    # For "2 AI reviewers": for each human item, check if covered by >=1
    #   of any pair of AI reviewers. Average across all C(3,2)=3 pairs.
    # For "3 AI reviewers": for each human item, check if covered by any
    #   of the 3 AI reviewers.

    all_items_1ai = []     # per individual AI reviewer coverage fractions
    all_items_2ai = []     # per pair-of-AI coverage fractions
    all_items_3ai = []     # all-3-AI coverage fractions
    fg_items_1ai = []
    fg_items_2ai = []
    fg_items_3ai = []
    ai_covered_fracs = []  # AI items covered by at least one human

    for pid in paper_ids:
        h_items = human_items_by_paper.get(pid, [])
        a_items = ai_items_by_paper.get(pid, [])
        coverage_map = h_covered_by_ai.get(pid, {})

        if not h_items:
            continue

        # For each human item, which AI reviewers cover it?
        item_ai_sets = {}
        for (h_rev, h_item) in h_items:
            item_ai_sets[(h_rev, h_item)] = coverage_map.get((h_rev, h_item), set())

        fg_items = [(h_rev, h_item) for (h_rev, h_item) in h_items
                    if (pid, h_rev, h_item) in fully_good_set]

        # 1 AI reviewer: average over each individual AI reviewer
        for ai_r in AI_REVIEWERS:
            if not h_items:
                continue
            covered = sum(1 for k in h_items if ai_r in item_ai_sets[k])
            all_items_1ai.append(covered / len(h_items))
            if fg_items:
                fg_covered = sum(1 for k in fg_items if ai_r in item_ai_sets[k])
                fg_items_1ai.append(fg_covered / len(fg_items))

        # 2 AI reviewers: average over all pairs
        for i in range(len(AI_REVIEWERS)):
            for j in range(i + 1, len(AI_REVIEWERS)):
                pair = {AI_REVIEWERS[i], AI_REVIEWERS[j]}
                covered = sum(1 for k in h_items if item_ai_sets[k] & pair)
                all_items_2ai.append(covered / len(h_items))
                if fg_items:
                    fg_covered = sum(1 for k in fg_items if item_ai_sets[k] & pair)
                    fg_items_2ai.append(fg_covered / len(fg_items))

        # 3 AI reviewers: any of the 3
        all_ai = set(AI_REVIEWERS)
        covered = sum(1 for k in h_items if item_ai_sets[k] & all_ai)
        all_items_3ai.append(covered / len(h_items))
        if fg_items:
            fg_covered = sum(1 for k in fg_items if item_ai_sets[k] & all_ai)
            fg_items_3ai.append(fg_covered / len(fg_items))

        # AI items covered by at least one human
        if a_items:
            ai_cov_map = ai_covered_by_h.get(pid, {})
            covered_ai = sum(1 for k in a_items if ai_cov_map.get(k, False))
            ai_covered_fracs.append(covered_ai / len(a_items))

    results = {
        "1ai_all": np.mean(all_items_1ai) if all_items_1ai else 0.0,
        "2ai_all": np.mean(all_items_2ai) if all_items_2ai else 0.0,
        "3ai_all": np.mean(all_items_3ai) if all_items_3ai else 0.0,
        "1ai_fg": np.mean(fg_items_1ai) if fg_items_1ai else 0.0,
        "2ai_fg": np.mean(fg_items_2ai) if fg_items_2ai else 0.0,
        "3ai_fg": np.mean(fg_items_3ai) if fg_items_3ai else 0.0,
        "ai_by_human": np.mean(ai_covered_fracs) if ai_covered_fracs else 0.0,
    }
    return results


def compute_paper_stats_for_bootstrap(paper_ids, human_items_by_paper, fully_good_set,
                                      h_covered_by_ai, ai_covered_by_h, ai_items_by_paper):
    """Same as compute_paper_stats but aggregates per-human-reviewer-instance fractions
    across the (possibly resampled) paper_ids."""

    # Collect per-(paper, human_reviewer) coverage fractions
    all_1ai = []
    all_2ai = []
    all_3ai = []
    fg_1ai = []
    fg_2ai = []
    fg_3ai = []
    ai_by_h = []

    for pid in paper_ids:
        h_items = human_items_by_paper.get(pid, [])
        a_items = ai_items_by_paper.get(pid, [])
        coverage_map = h_covered_by_ai.get(pid, {})

        if not h_items:
            continue

        # Group human items by human reviewer
        by_h_rev = defaultdict(list)
        for (h_rev, h_item) in h_items:
            by_h_rev[h_rev].append((h_rev, h_item))

        for h_rev, items in by_h_rev.items():
            item_ai_sets = {k: coverage_map.get(k, set()) for k in items}
            fg = [k for k in items if (pid, k[0], k[1]) in fully_good_set]

            # 1 AI reviewer: average over the 3 individual AI reviewers
            for ai_r in AI_REVIEWERS:
                cov = sum(1 for k in items if ai_r in item_ai_sets[k])
                all_1ai.append(cov / len(items))
                if fg:
                    fg_cov = sum(1 for k in fg if ai_r in item_ai_sets[k])
                    fg_1ai.append(fg_cov / len(fg))

            # 2 AI reviewers: average over all C(3,2) pairs
            for i in range(len(AI_REVIEWERS)):
                for j in range(i + 1, len(AI_REVIEWERS)):
                    pair = {AI_REVIEWERS[i], AI_REVIEWERS[j]}
                    cov = sum(1 for k in items if item_ai_sets[k] & pair)
                    all_2ai.append(cov / len(items))
                    if fg:
                        fg_cov = sum(1 for k in fg if item_ai_sets[k] & pair)
                        fg_2ai.append(fg_cov / len(fg))

            # 3 AI reviewers
            all_ai = set(AI_REVIEWERS)
            cov = sum(1 for k in items if item_ai_sets[k] & all_ai)
            all_3ai.append(cov / len(items))
            if fg:
                fg_cov = sum(1 for k in fg if item_ai_sets[k] & all_ai)
                fg_3ai.append(fg_cov / len(fg))

        # AI items covered by at least one human
        if a_items:
            ai_cov_map = ai_covered_by_h.get(pid, {})
            covered_ai = sum(1 for k in a_items if ai_cov_map.get(k, False))
            ai_by_h.append(covered_ai / len(a_items))

    return {
        "1ai_all": np.mean(all_1ai) if all_1ai else 0.0,
        "2ai_all": np.mean(all_2ai) if all_2ai else 0.0,
        "3ai_all": np.mean(all_3ai) if all_3ai else 0.0,
        "1ai_fg": np.mean(fg_1ai) if fg_1ai else 0.0,
        "2ai_fg": np.mean(fg_2ai) if fg_2ai else 0.0,
        "3ai_fg": np.mean(fg_3ai) if fg_3ai else 0.0,
        "ai_by_human": np.mean(ai_by_h) if ai_by_h else 0.0,
    }


# ── paper-level bootstrap ───────────────────────────────────────────
def bootstrap_ci(human_items_by_paper, fully_good_set, h_covered_by_ai,
                 ai_covered_by_h, ai_items_by_paper, n_boot=N_BOOT, seed=SEED):
    all_paper_ids = sorted(human_items_by_paper.keys())
    n_papers = len(all_paper_ids)
    paper_arr = np.array(all_paper_ids)
    rng = np.random.RandomState(seed)

    # Observed statistics
    obs = compute_paper_stats_for_bootstrap(
        all_paper_ids, human_items_by_paper, fully_good_set,
        h_covered_by_ai, ai_covered_by_h, ai_items_by_paper
    )

    metric_names = list(obs.keys())
    boot_values = {m: np.empty(n_boot) for m in metric_names}

    for b in range(n_boot):
        sampled = rng.choice(paper_arr, size=n_papers, replace=True)
        stats = compute_paper_stats_for_bootstrap(
            sampled, human_items_by_paper, fully_good_set,
            h_covered_by_ai, ai_covered_by_h, ai_items_by_paper
        )
        for m in metric_names:
            boot_values[m][b] = stats[m]

    cis = {}
    for m in metric_names:
        cis[m] = (np.percentile(boot_values[m], 2.5),
                  np.percentile(boot_values[m], 97.5))

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

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    x_labels = ["1 AI reviewer", "2 AI reviewers", "3 AI reviewers"]
    x = np.arange(len(x_labels))
    width = 0.30

    # "All human items" group
    all_vals = [obs["1ai_all"], obs["2ai_all"], obs["3ai_all"]]
    all_errs_lo = [obs["1ai_all"] - cis["1ai_all"][0],
                   obs["2ai_all"] - cis["2ai_all"][0],
                   obs["3ai_all"] - cis["3ai_all"][0]]
    all_errs_hi = [cis["1ai_all"][1] - obs["1ai_all"],
                   cis["2ai_all"][1] - obs["2ai_all"],
                   cis["3ai_all"][1] - obs["3ai_all"]]

    # "Fully good items only" group
    fg_vals = [obs["1ai_fg"], obs["2ai_fg"], obs["3ai_fg"]]
    fg_errs_lo = [obs["1ai_fg"] - cis["1ai_fg"][0],
                  obs["2ai_fg"] - cis["2ai_fg"][0],
                  obs["3ai_fg"] - cis["3ai_fg"][0]]
    fg_errs_hi = [cis["1ai_fg"][1] - obs["1ai_fg"],
                  cis["2ai_fg"][1] - obs["2ai_fg"],
                  cis["3ai_fg"][1] - obs["3ai_fg"]]

    bars1 = ax.bar(x - width / 2, [v * 100 for v in all_vals], width,
                   yerr=[[e * 100 for e in all_errs_lo],
                         [e * 100 for e in all_errs_hi]],
                   capsize=3, color="#0072B2", edgecolor="white", linewidth=0.3,
                   label="All human items", error_kw={"linewidth": 0.8})

    bars2 = ax.bar(x + width / 2, [v * 100 for v in fg_vals], width,
                   yerr=[[e * 100 for e in fg_errs_lo],
                         [e * 100 for e in fg_errs_hi]],
                   capsize=3, color="#E69F00", edgecolor="white", linewidth=0.3,
                   label="Fully good items only", error_kw={"linewidth": 0.8})

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                    f"{height:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Coverage (%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nWrote {OUT_PDF}")


# ── main ─────────────────────────────────────────────────────────────
def main():
    pairs = load_pairs(DATA_FILE)
    annotations = load_annotations()

    human_items_by_paper, fully_good_set, h_covered_by_ai, \
        ai_covered_by_h, ai_items_by_paper = build_coverage(pairs, annotations)

    print(f"\nPapers: {len(human_items_by_paper)}")
    total_h = sum(len(v) for v in human_items_by_paper.values())
    total_a = sum(len(v) for v in ai_items_by_paper.values())
    print(f"Total human items: {total_h}")
    print(f"Total AI items: {total_a}")
    total_fg = sum(1 for pid, items in human_items_by_paper.items()
                   for (h_rev, h_item) in items
                   if (pid, h_rev, h_item) in fully_good_set)
    print(f"Fully good human items: {total_fg}")

    obs, cis = bootstrap_ci(human_items_by_paper, fully_good_set,
                            h_covered_by_ai, ai_covered_by_h, ai_items_by_paper)

    print("\n=== Coverage Results ===")
    labels = {
        "1ai_all": "Human items covered by 1 AI reviewer (avg)",
        "2ai_all": "Human items covered by 2 AI reviewers (avg)",
        "3ai_all": "Human items covered by 3 AI reviewers",
        "1ai_fg": "Fully-good human items covered by 1 AI (avg)",
        "2ai_fg": "Fully-good human items covered by 2 AI (avg)",
        "3ai_fg": "Fully-good human items covered by 3 AI",
        "ai_by_human": "AI items covered by >=1 human reviewer",
    }
    for m, label in labels.items():
        ci = cis[m]
        print(f"  {label}: {obs[m]*100:.1f}%  95% CI [{ci[0]*100:.1f}, {ci[1]*100:.1f}]")

    make_figure(obs, cis)


if __name__ == "__main__":
    main()
