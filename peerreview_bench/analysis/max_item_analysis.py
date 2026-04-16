#!/usr/bin/env python3
"""
Max-items hyperparameter analysis.

The AI reviewer agents are instructed to write *at most 5* review items per
paper, ranked by importance (item 1 = most important). This script simulates
the effect of altering that cap to k = 1, 2, 3, 4, 5 by truncating the
per-(paper, reviewer) item list to items where `review_item_number <= k`,
then recomputing the three core metrics (correctness, significance mean,
evidence sufficiency) plus the composite "fully good" rate for each AI model.

Cascade stripping caveat
------------------------
Some items written by the reviewer were dropped during HF upload because
they were not fully annotated (rule 2: missing significance; rule 3:
missing evidence). Those gaps are preserved as-is: if a reviewer wrote
items 1, 2, 3, 4, 5 but item 3 was dropped, then at k=3 we keep only items
1 and 2 (since item 3's review_item_number <= 3 matches nothing in the
HF data), and at k=5 we keep items 1, 2, 4, 5. This is the right behavior:
the truncation is applied to the *published* items, which is what a future
pipeline with a tighter cap would see.

Outputs
-------
  analysis/main_results/figure3.svg
  analysis/main_results/figure3.png
  (also prints a terminal-friendly table)
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Allow `load_data` / `data_filter` imports from the sibling dirs
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # peerreview_bench/
sys.path.insert(0, str(_HERE))         # peerreview_bench/analysis/

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from data_filter import (
    get_items_for_correctness, get_items_for_significance, get_items_for_evidence,
    is_fully_good,
)
from load_data import load_annotations, ReviewItem


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MAX_ITEMS_RANGE = [1, 2, 3, 4, 5]
AI_MODELS = ['GPT', 'Claude', 'Gemini']
HUMAN_GROUPS = ['Top-Rated Human', 'Lowest-Rated Human']

# Colors consistent with the paper's visual identity
MODEL_COLORS = {
    'GPT':    '#1f77b4',
    'Claude': '#d62728',
    'Gemini': '#2ca02c',
    'Top-Rated Human':    '#555555',
    'Lowest-Rated Human': '#999999',
}

OUTPUT_DIR = _HERE / 'main_results'
FIG_SVG = OUTPUT_DIR / 'figure3.svg'
FIG_PNG = OUTPUT_DIR / 'figure3.png'

RANDOM_SEED = 42
N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# GROUP CONSTRUCTION (mirrors peerreview_analysis.py::create_comparison_groups)
# ---------------------------------------------------------------------------

def create_comparison_groups(items: List[ReviewItem],
                              rankings: Dict[int, Dict[str, str]]
                              ) -> Dict[str, List[ReviewItem]]:
    groups: Dict[str, List[ReviewItem]] = {
        'Top-Rated Human': [], 'Lowest-Rated Human': [],
        'GPT': [], 'Claude': [], 'Gemini': [],
    }
    for it in items:
        pid = it.paper_id
        if it.reviewer_type == 'Human':
            if pid not in rankings:
                continue
            r = rankings[pid]
            if it.reviewer_id == r.get('best'):
                groups['Top-Rated Human'].append(it)
            # intentionally not elif — when best == worst, the same item
            # goes into both groups (see peerreview_analysis.py).
            if it.reviewer_id == r.get('worst'):
                groups['Lowest-Rated Human'].append(it)
        elif it.reviewer_type == 'AI':
            if it.model_name in groups:
                groups[it.model_name].append(it)
    return groups


# ---------------------------------------------------------------------------
# METRIC HELPERS
# ---------------------------------------------------------------------------

def _paper_rates_binary(items: List[ReviewItem], metric: str) -> List[float]:
    """Per-paper fraction (binary metrics) aggregating over items.

    `metric` in {'correctness', 'evidence', 'fully_good'}. We key by
    paper_id alone — if the item list spans primary+secondary annotators
    on an overlap paper, their items are pooled for that paper (consistent
    with the main analysis pipeline).
    """
    if metric == 'correctness':
        filtered = get_items_for_correctness(items)
        get_v = lambda i: i.correctness_numeric
    elif metric == 'evidence':
        filtered = get_items_for_evidence(items)
        get_v = lambda i: i.evidence_numeric
    elif metric == 'fully_good':
        # Fully good rate uses the full (non-cascade-stripped) item set,
        # counting only items that are Correct + Significant + Sufficient.
        filtered = items
        get_v = lambda i: 1 if is_fully_good(i) else 0
    else:
        raise ValueError(f"Unknown binary metric: {metric}")

    by_paper: Dict[int, List[int]] = defaultdict(list)
    for it in filtered:
        by_paper[it.paper_id].append(get_v(it))
    return [sum(v) / len(v) for v in by_paper.values() if v]


def _paper_rates_ordinal(items: List[ReviewItem]) -> List[float]:
    """Per-paper mean significance (0-2 scale, ordinal)."""
    filtered = get_items_for_significance(items)
    by_paper: Dict[int, List[int]] = defaultdict(list)
    for it in filtered:
        by_paper[it.paper_id].append(it.significance_numeric)
    return [float(np.mean(v)) for v in by_paper.values() if v]


def _t_ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) via student-t interval."""
    n = len(values)
    if n == 0:
        return (float('nan'),) * 3
    if n == 1:
        return (float(values[0]), float(values[0]), float(values[0]))
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    se = float(sp_stats.sem(arr, ddof=1))
    half = float(sp_stats.t.ppf(1 - alpha / 2, df=n - 1)) * se
    return m, m - half, m + half


def _bootstrap_ci(values: List[float], alpha: float = 0.05,
                  n_boot: int = N_BOOTSTRAP) -> Tuple[float, float, float]:
    """Non-parametric bootstrap mean CI — used for the ordinal significance
    metric, to match the per-paper CI convention in Table 1."""
    n = len(values)
    if n == 0:
        return (float('nan'),) * 3
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    if n == 1:
        return m, m, m
    rng = np.random.default_rng(RANDOM_SEED)
    boots = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    return m, float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


# ---------------------------------------------------------------------------
# CORE COMPUTATION
# ---------------------------------------------------------------------------

def compute_metrics_by_k(groups: Dict[str, List[ReviewItem]],
                          ks: List[int]) -> Dict[str, Dict[int, Dict[str, Tuple[float, float, float]]]]:
    """For each AI model and k in ks, compute the four main metrics on
    items with review_item_number <= k. Human groups are computed once at
    k = max(ks) since max_items doesn't apply to them (humans are not
    constrained to a 5-item cap).

    Returns a nested dict:
      out[group_name][k][metric] = (mean, ci_lo, ci_hi)
    Item counts are stored at out[group_name][k]['_n_items'] = int.
    """
    out: Dict[str, Dict[int, Dict[str, Tuple[float, float, float]]]] = {}

    for name, items in groups.items():
        out[name] = {}
        is_ai = name not in HUMAN_GROUPS
        for k in ks:
            if is_ai:
                truncated = [it for it in items if it.item_number <= k]
            else:
                # Humans are not affected by the AI max_items cap.
                truncated = items
            c = _paper_rates_binary(truncated, 'correctness')
            s = _paper_rates_ordinal(truncated)
            e = _paper_rates_binary(truncated, 'evidence')
            fg = _paper_rates_binary(truncated, 'fully_good')
            out[name][k] = {
                'correctness': _t_ci(c),
                'significance': _bootstrap_ci(s),
                'evidence': _t_ci(e),
                'fully_good': _t_ci(fg),
                '_n_items': len(truncated),
                '_n_papers': len({it.paper_id for it in truncated}),
            }
    return out


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_figure(metrics: Dict, ks: List[int]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 12.5,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 150,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), sharex=True)

    # Per-panel config: (metric_key, title, unit, y_min, y_max, ax)
    panels = [
        ('correctness',  'Correctness rate',         '%',   65.0, 100.0, axes[0, 0]),
        ('significance', 'Significance mean',        '0–2', 1.20, 1.90, axes[0, 1]),
        ('evidence',     'Evidence sufficiency rate', '%',  82.0, 100.0, axes[1, 0]),
        ('fully_good',   'Fully good rate',          '%',   30.0, 80.0, axes[1, 1]),
    ]

    for metric_key, title, unit, y_min, y_max, ax in panels:
        y_scale = 100.0 if unit == '%' else 1.0
        # Clip helper that caps CI bands at the physical ceiling (100%)
        clip_hi = 100.0 if unit == '%' else float('inf')
        clip_lo = 0.0 if unit == '%' else -float('inf')

        # AI model lines
        for model in AI_MODELS:
            means, lows, highs = [], [], []
            for k in ks:
                m, lo, hi = metrics[model][k][metric_key]
                means.append(m * y_scale)
                lows.append(max(lo * y_scale, clip_lo))
                highs.append(min(hi * y_scale, clip_hi))
            ax.plot(ks, means, marker='o', linewidth=2.2, markersize=7,
                    color=MODEL_COLORS[model], label=model, zorder=5)
            ax.fill_between(ks, lows, highs, alpha=0.16, color=MODEL_COLORS[model],
                            zorder=3, linewidth=0)

        # Human reference lines (constant across k)
        k_ref = ks[-1]
        for human_name, line_style in (
            ('Top-Rated Human', '--'),
            ('Lowest-Rated Human', ':'),
        ):
            m, _, _ = metrics[human_name][k_ref][metric_key]
            ax.axhline(m * y_scale,
                       color=MODEL_COLORS[human_name],
                       linestyle=line_style,
                       linewidth=1.6,
                       label=human_name,
                       zorder=2)

        ax.set_title(title, pad=8)
        if unit == '%':
            ax.set_ylabel(f'{title} (%)')
        else:
            ax.set_ylabel(f'{title}')
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - 0.25, ks[-1] + 0.25)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.25, linestyle=':')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    # Shared x-axis label
    for col in range(2):
        axes[1, col].set_xlabel('max_items cap (truncate items with review_item_number > k)')

    # Shared legend at the bottom — ordered AI models then humans
    order = AI_MODELS + HUMAN_GROUPS
    handles_labels = axes[0, 0].get_legend_handles_labels()
    h_map = dict(zip(handles_labels[1], handles_labels[0]))
    ordered = [(h_map[l], l) for l in order if l in h_map]
    fig.legend(
        [u[0] for u in ordered],
        [u[1] for u in ordered],
        loc='lower center',
        ncol=len(ordered),
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    fig.savefig(FIG_SVG, format='svg', bbox_inches='tight')
    fig.savefig(FIG_PNG, format='png', bbox_inches='tight', dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# TERMINAL REPORT
# ---------------------------------------------------------------------------

def print_table(metrics: Dict, ks: List[int]):
    print()
    print("=" * 96)
    print("MAX_ITEMS HYPERPARAMETER SWEEP — PER-PAPER MEAN (t/bootstrap 95% CI)")
    print("=" * 96)
    header = f"  {'Group':<20} " + " ".join(f"k={k}".rjust(18) for k in ks) + "  n_items@k=5"
    for metric_key, pretty in [('correctness', 'Correctness (%)'),
                               ('significance', 'Significance 0-2'),
                               ('evidence', 'Evidence (%)'),
                               ('fully_good', 'Fully good (%)')]:
        print(f"\n-- {pretty} --")
        print(header)
        for group in AI_MODELS + HUMAN_GROUPS:
            cells = []
            for k in ks:
                m, lo, hi = metrics[group][k][metric_key]
                if metric_key == 'significance':
                    cell = f"{m:4.2f}[{lo:.2f},{hi:.2f}]"
                else:
                    cell = f"{m*100:5.1f}[{lo*100:4.1f},{hi*100:4.1f}]"
                cells.append(cell.rjust(18))
            n5 = metrics[group][ks[-1]]['_n_items']
            print(f"  {group:<20} " + " ".join(cells) + f"   {n5}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading annotations from HuggingFace (expert_annotation)...")
    items, rankings = load_annotations('both')
    print(f"  loaded {len(items):,} items, {len(rankings)} papers with rankings")

    groups = create_comparison_groups(items, rankings)
    for name, its in groups.items():
        pids = {i.paper_id for i in its}
        print(f"  group={name:<20} n_items={len(its):>4}  n_papers={len(pids)}")

    metrics = compute_metrics_by_k(groups, MAX_ITEMS_RANGE)
    print_table(metrics, MAX_ITEMS_RANGE)

    plot_figure(metrics, MAX_ITEMS_RANGE)
    print(f"  -> wrote {FIG_SVG}")
    print(f"  -> wrote {FIG_PNG}")


if __name__ == '__main__':
    main()
