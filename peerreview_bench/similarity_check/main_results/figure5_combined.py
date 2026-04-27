#!/usr/bin/env python3
"""
Generate figure5_overlap.{pdf,png} — combined overlap analysis figure.

Left panel:  4-category similarity distribution (HH, HA, AA)
Right panel: Coverage by reviewer configuration (1H→1H, 1H→1AI, 3H→3AI)

All values are Rogan-Gladen-corrected prevalences using the GPT-5.4
similarity judge calibration (sensitivity=87.1%, specificity=96.8%
on the 164-pair eval set). CIs are 95% cluster-bootstrap (10,000
paper-level resamples).
"""

import json
import os
import sys
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
OUT_PDF = os.path.join(OUT_DIR, "figure5_overlap.pdf")
OUT_PNG = os.path.join(OUT_DIR, "figure5_overlap.png")
OUT_SVG = os.path.join(OUT_DIR, "figure5_overlap.svg")

DROPPED_PAPER_IDS = {11, 20, 22}
N_BOOT = 10_000
SEED = 42

from rogan_gladen import rogan_gladen_correct, resample_sens_spec, SENSITIVITY, SPECIFICITY


# ── 4-way ordinal mapping ─────────────────────────────────────────────
NEAR_PARAPHRASE = "same subject, same argument, same evidence"
CONVERGENT = "same subject, same argument, different evidence"
TOPICAL = "same subject, different argument"
UNRELATED = "different subject"

CATEGORY_ORDER = [UNRELATED, TOPICAL, CONVERGENT, NEAR_PARAPHRASE]  # bottom to top

ORDINAL = {
    NEAR_PARAPHRASE: 3,
    CONVERGENT: 2,
    TOPICAL: 1,
    UNRELATED: 0,
}

# ── colors (matched to §2.4 category box styles) ─────────────────────
# Left panel: red-to-green gradient (dissimilar → identical)
COLORS_LEFT = {
    UNRELATED:       "#B4463C",   # muted terracotta (most dissimilar)
    TOPICAL:         "#D89990",   # light red/pink
    CONVERGENT:      "#A8C7AC",   # light green
    NEAR_PARAPHRASE: "#3C784B",   # muted dark green (most identical)
}

# Right panel: reuses left-panel colors for visual correspondence
# Broadest threshold → strictest threshold
COLORS_RIGHT = ["#D89990", "#A8C7AC", "#3C784B"]


# ── load ───────────────────────────────────────────────────────────────
def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("paper_id") in DROPPED_PAPER_IDS:
                continue
            if r.get("error") is not None and str(r["error"]) != "None":
                continue
            pairs.append(r)
    print(f"Loaded {len(pairs)} valid pairs.")
    return pairs


# ── Panel 1: 4-category distribution ──────────────────────────────────
def compute_panel1_data(pairs):
    """Compute per-pair-type 4-category distribution with Rogan-Gladen
    correction and paper-level bootstrap CIs."""

    bar_filters = [
        ("Human vs Human", lambda p: p["pair_type"] == "H-H" and not p["same_reviewer"]),
        ("Human vs AI",    lambda p: p["pair_type"] == "H-A"),
        ("AI vs AI",       lambda p: p["pair_type"] == "A-A" and not p["same_reviewer"]),
    ]

    rng = np.random.RandomState(SEED)
    results = {}

    for label, filt in bar_filters:
        subset = [p for p in pairs if filt(p)]

        # Group by paper
        by_paper = defaultdict(list)
        for p in subset:
            by_paper[p["paper_id"]].append(p)

        paper_ids = sorted(by_paper.keys())
        n_papers = len(paper_ids)

        # Compute observed fractions
        n = len(subset)
        counts = defaultdict(int)
        for p in subset:
            counts[p["parsed_answer"]] += 1

        # Raw observed similar rate (for Rogan-Gladen)
        obs_similar = (counts[NEAR_PARAPHRASE] + counts[CONVERGENT]) / n if n > 0 else 0
        corrected_similar = rogan_gladen_correct(obs_similar)

        # Distribute correction proportionally within similar/not-similar
        obs_fracs = {cat: counts[cat] / n if n > 0 else 0 for cat in CATEGORY_ORDER}

        # Similar categories
        sim_total_obs = obs_fracs[NEAR_PARAPHRASE] + obs_fracs[CONVERGENT]
        if sim_total_obs > 0:
            np_ratio = obs_fracs[NEAR_PARAPHRASE] / sim_total_obs
            conv_ratio = obs_fracs[CONVERGENT] / sim_total_obs
        else:
            np_ratio = conv_ratio = 0.5

        corrected_fracs = {}
        corrected_fracs[NEAR_PARAPHRASE] = corrected_similar * np_ratio
        corrected_fracs[CONVERGENT] = corrected_similar * conv_ratio

        # Not-similar categories
        not_sim_total_obs = obs_fracs[TOPICAL] + obs_fracs[UNRELATED]
        corrected_not_similar = 1.0 - corrected_similar
        if not_sim_total_obs > 0:
            top_ratio = obs_fracs[TOPICAL] / not_sim_total_obs
            unr_ratio = obs_fracs[UNRELATED] / not_sim_total_obs
        else:
            top_ratio = unr_ratio = 0.5

        corrected_fracs[TOPICAL] = corrected_not_similar * top_ratio
        corrected_fracs[UNRELATED] = corrected_not_similar * unr_ratio

        # Bootstrap CIs (paper-level) at each cumulative level
        # Cumulative levels (bottom to top): UNRELATED, +TOPICAL, +CONVERGENT, +NEAR_PARA
        # We compute CIs on the cumulative fraction at each boundary
        cumulative_thresholds = [
            ("unrelated",  {UNRELATED}),
            ("+topical",   {UNRELATED, TOPICAL}),
            ("+convergent",{UNRELATED, TOPICAL, CONVERGENT}),
            ("+near_para", {UNRELATED, TOPICAL, CONVERGENT, NEAR_PARAPHRASE}),
        ]

        boot_cumulative = {name: np.empty(N_BOOT) for name, _ in cumulative_thresholds}
        for b in range(N_BOOT):
            sampled_pids = rng.choice(paper_ids, size=n_papers, replace=True)
            boot_pairs = []
            for pid in sampled_pids:
                boot_pairs.extend(by_paper[pid])
            bn = len(boot_pairs)
            if bn == 0:
                for name, _ in cumulative_thresholds:
                    boot_cumulative[name][b] = 0
                continue
            b_counts = defaultdict(int)
            for p in boot_pairs:
                b_counts[p["parsed_answer"]] += 1
            # Resample sens/spec and apply Rogan-Gladen
            b_sens, b_spec = resample_sens_spec(rng)
            b_sim = (b_counts[NEAR_PARAPHRASE] + b_counts[CONVERGENT]) / bn
            b_corr_sim = rogan_gladen_correct(b_sim, b_sens, b_spec)
            b_not_sim = 1.0 - b_corr_sim
            # Proportional split within similar
            sim_tot = b_counts[NEAR_PARAPHRASE] + b_counts[CONVERGENT]
            np_r = b_counts[NEAR_PARAPHRASE] / sim_tot if sim_tot > 0 else 0.5
            # Proportional split within not-similar
            nsim_tot = b_counts[TOPICAL] + b_counts[UNRELATED]
            top_r = b_counts[TOPICAL] / nsim_tot if nsim_tot > 0 else 0.5

            b_fracs = {
                UNRELATED: b_not_sim * (1 - top_r),
                TOPICAL: b_not_sim * top_r,
                CONVERGENT: b_corr_sim * (1 - np_r),
                NEAR_PARAPHRASE: b_corr_sim * np_r,
            }
            cum = 0
            for name, cat_set in cumulative_thresholds:
                cum = sum(b_fracs[c] for c in cat_set)
                boot_cumulative[name][b] = cum

        ci_cumulative = {}
        for name, _ in cumulative_thresholds:
            ci_cumulative[name] = (np.percentile(boot_cumulative[name], 2.5),
                                   np.percentile(boot_cumulative[name], 97.5))

        results[label] = {
            'fracs': corrected_fracs,
            'similar_rate': corrected_similar,
            'ci_cumulative': ci_cumulative,
            'n_pairs': n,
        }

        print(f"  {label}: n={n}, similar={corrected_similar*100:.1f}%")
        for cat in CATEGORY_ORDER:
            print(f"    {cat[:30]:30s}: {corrected_fracs[cat]*100:.1f}%")

    return results


# ── Panel 2: Coverage ─────────────────────────────────────────────────
def build_similarity_index(pairs):
    """Build: (paper_id, source_reviewer, source_item, target_reviewer) -> max ordinal."""
    sim = defaultdict(lambda: 0)
    for p in pairs:
        if p["pair_type"] not in ("H-A", "H-H"):
            continue
        if p["pair_type"] == "H-H" and p["same_reviewer"]:
            continue
        ov = ORDINAL[p["parsed_answer"]]
        a, b = p["item_a"], p["item_b"]
        key_ab = (p["paper_id"], a["reviewer_id"], a["review_item_number"], b["reviewer_id"])
        sim[key_ab] = max(sim[key_ab], ov)
        key_ba = (p["paper_id"], b["reviewer_id"], b["review_item_number"], a["reviewer_id"])
        sim[key_ba] = max(sim[key_ba], ov)
    return dict(sim)


def get_items_by_paper(pairs):
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


def coverage_1to1(sim_index, items_by_paper, source_type, target_type, threshold):
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
    return by_paper


def coverage_all_to_all(sim_index, items_by_paper, source_type, target_type, threshold):
    """For each source reviewer, compute coverage by the union of all target reviewers.
    Then average across source reviewers within each paper.
    by_paper[pid] = [mean coverage across source reviewers]."""
    by_paper = defaultdict(list)
    for pid, items in items_by_paper.items():
        source_reviewers = set(rid for rid, _, rtype in items if rtype == source_type)
        target_reviewers = set(rid for rid, _, rtype in items if rtype == target_type)
        for srev in source_reviewers:
            source_items = [(rid, inum) for rid, inum, rtype in items
                            if rtype == source_type and rid == srev]
            covered = 0
            for srid, sinum in source_items:
                max_sim = 0
                for trev in target_reviewers:
                    s = sim_index.get((pid, srid, sinum, trev), 0)
                    max_sim = max(max_sim, s)
                if max_sim >= threshold:
                    covered += 1
            if source_items:
                by_paper[pid].append(covered / len(source_items))
    return by_paper


def paper_bootstrap_corrected(by_paper, n_boot=N_BOOT, seed=SEED):
    """Paper-level bootstrap CI with Rogan-Gladen correction.
    Each paper contributes one value (mean of its entries in by_paper)."""
    paper_ids = sorted(by_paper.keys())
    n_papers = len(paper_ids)
    rng = np.random.RandomState(seed)

    # Observed: paper-level mean
    paper_means = [np.mean(by_paper[pid]) for pid in paper_ids]
    obs = np.mean(paper_means) if paper_means else 0.0
    corrected_obs = rogan_gladen_correct(obs)

    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(paper_ids, size=n_papers, replace=True)
        b_means = [np.mean(by_paper[pid]) for pid in sampled]
        raw = np.mean(b_means) if b_means else 0.0
        b_sens, b_spec = resample_sens_spec(rng)
        boot[b] = rogan_gladen_correct(raw, b_sens, b_spec)

    return corrected_obs, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def compute_panel2_data(pairs):
    sim_index = build_similarity_index(pairs)
    items_by_paper = get_items_by_paper(pairs)

    thresholds = [1, 2, 3]  # same issue, same criticism, same evidence
    threshold_labels = ["At least\nsame issue", "At least\nsame criticism",
                        "Exact same\nevidence"]

    configs = [
        ("1 Human\n→ 1 Human", "1to1_hh", "Human", "Human"),
        ("1 Human\n→ 1 AI",    "1to1",    "Human", "AI"),
        ("3 Humans\n→ 3 AIs",  "all",     "Human", "AI"),
    ]

    data = {}
    for g_label, mode, src, tgt in configs:
        for t in thresholds:
            if mode == "1to1" or mode == "1to1_hh":
                bp = coverage_1to1(sim_index, items_by_paper, src, tgt, t)
            else:
                bp = coverage_all_to_all(sim_index, items_by_paper, src, tgt, t)
            obs, lo, hi = paper_bootstrap_corrected(bp)
            data[(g_label, t)] = (obs, lo, hi)
            print(f"  {g_label.replace(chr(10), ' ')} @ threshold={t}: "
                  f"{obs*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]")

    return data, configs, thresholds, threshold_labels


# ── plot ───────────────────────────────────────────────────────────────
def make_figure(pairs):
    print("\n=== Panel 1: 4-category distribution ===")
    panel1 = compute_panel1_data(pairs)

    print("\n=== Panel 2: Coverage ===")
    panel2_data, configs, thresholds, threshold_labels = compute_panel2_data(pairs)

    # ── figure setup ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 12,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [1.1, 1.0]})

    # ── Panel 1: Stacked bar ─────────────────────────────────────────
    bar_labels = list(panel1.keys())
    x1 = np.arange(len(bar_labels))
    width1 = 0.55

    bottoms = np.zeros(len(bar_labels))

    # Cumulative level names matching bootstrap keys (bottom to top)
    cum_names = ["unrelated", "+topical", "+convergent", "+near_para"]

    for cat_idx, cat in enumerate(CATEGORY_ORDER):
        vals = np.array([panel1[label]['fracs'][cat] for label in bar_labels])
        ax1.bar(x1, vals, width1, bottom=bottoms, color=COLORS_LEFT[cat],
                edgecolor="white", linewidth=0.5)

        # In-bar annotations
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v < 0.005:
                continue
            pct_str = f"{v*100:.1f}%"
            y_center = b + v / 2
            text_color = ("white" if cat in (NEAR_PARAPHRASE, UNRELATED)
                          else "#333333")
            if v < 0.015:
                continue  # too small for any label
            elif v < 0.04:
                # Small segment: plain text above the bar, no connector
                ax1.text(x1[i], b + v + 0.008, pct_str,
                         ha="center", va="bottom", fontsize=11, color="#333333",
                         fontweight="bold")
            else:
                ax1.text(x1[i], y_center, pct_str, ha="center", va="center",
                         fontsize=13, color=text_color, fontweight="bold")

        bottoms += vals

    ax1.set_ylabel("Fraction of review item pairs (%)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(bar_labels, fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.tick_params(axis="y", labelsize=12)
    ax1.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Legend for panel 1 — below the panel, horizontal
    legend_cats = [NEAR_PARAPHRASE, CONVERGENT, TOPICAL, UNRELATED]
    legend_labels = [
        "Same issue, same criticism,\nsame evidence",
        "Same issue, same criticism,\ndifferent evidence",
        "Same issue,\ndifferent criticism",
        "Different\nissue",
    ]
    handles1 = [Patch(facecolor=COLORS_LEFT[cat], edgecolor="white", linewidth=0.5)
                for cat in legend_cats]

    leg1 = ax1.legend(handles1, legend_labels, loc="upper center",
                      bbox_to_anchor=(0.5, -0.10), ncol=4,
                      fontsize=12, frameon=True, fancybox=False,
                      edgecolor="#cccccc",
                      handletextpad=0.4, handlelength=1.2,
                      columnspacing=1.0)
    leg1.get_frame().set_alpha(0.95)

    # ── Panel 2: Grouped bars ────────────────────────────────────────
    group_labels = [gl for gl, _, _, _ in configs]
    n_groups = len(group_labels)
    n_bars = len(thresholds)
    bar_width = 0.22
    group_positions = np.arange(n_groups) * 1.0

    for i, t in enumerate(thresholds):
        offsets = group_positions + (i - 1) * bar_width
        vals = [panel2_data[(gl, t)][0] for gl in group_labels]
        lo_err = [panel2_data[(gl, t)][0] - panel2_data[(gl, t)][1] for gl in group_labels]
        hi_err = [panel2_data[(gl, t)][2] - panel2_data[(gl, t)][0] for gl in group_labels]

        ax2.bar(offsets, vals, bar_width, color=COLORS_RIGHT[i],
                edgecolor="white", linewidth=0.3,
                yerr=[lo_err, hi_err], capsize=3,
                error_kw={"linewidth": 0.8, "color": "black"},
                label=threshold_labels[i])

        # Annotations above bars
        for j, v in enumerate(vals):
            ax2.text(offsets[j], v + hi_err[j] + 0.015, f"{v*100:.1f}%",
                     ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax2.set_ylabel("Coverage (fraction of items)", fontsize=13, fontweight="bold")
    ax2.set_xticks(group_positions)
    ax2.set_xticklabels(group_labels, fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.tick_params(axis="y", labelsize=12)
    ax2.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    leg2 = ax2.legend(loc="upper left", fontsize=12, frameon=True,
                      fancybox=False, edgecolor="#cccccc",
                      labelspacing=0.5, handletextpad=0.4, handlelength=1.2)
    leg2.get_frame().set_alpha(0.95)

    # ── save ──────────────────────────────────────────────────────────
    fig.tight_layout(w_pad=3)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


# ── main ───────────────────────────────────────────────────────────────
def main():
    pairs = load_pairs(DATA_FILE)
    make_figure(pairs)


if __name__ == "__main__":
    main()
