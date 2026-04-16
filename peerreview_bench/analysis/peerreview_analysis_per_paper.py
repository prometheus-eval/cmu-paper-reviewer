#!/usr/bin/env python3
"""
PeerReview Bench - Per-Paper Paired Analysis

Paper-level aggregation with PAIRED tests (alternate framing of the primary
analysis). Reports per-group mean rates aggregated across papers.

- Binary metrics (correctness, evidence): paired t-test, Cohen's d
- Ordinal metric (significance, 0-2): Wilcoxon signed-rank, rank-biserial r
- Raw p-values (no multiple-comparison correction)

Usage:
    python peerreview_analysis_per_paper.py --data-dir <json_dir> --output-dir <output_dir> --rankings-file <rankings.json>
    python peerreview_analysis_per_paper.py --combined <combined.json> --output-dir <output_dir> --rankings-file <rankings.json>
"""

import json
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Allow `load_data` imports from the parent peerreview_bench/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_1samp, wilcoxon
from data_filter import (
    filter_kept_items,
    get_items_for_correctness as _filter_for_correctness,
    get_items_for_significance as _filter_for_significance,
    get_items_for_evidence as _filter_for_evidence,
    summarize_dataset, format_dataset_overview,
    compute_fully_good_stats, format_fully_good_report,
    is_fully_good,
)
from load_data import ReviewItem, load_annotations

warnings.filterwarnings('ignore')

ALPHA = 0.05

COHENS_D_THRESHOLDS = {'negligible': 0.20, 'small': 0.50, 'medium': 0.80}
RANK_BISERIAL_THRESHOLDS = {'negligible': 0.10, 'small': 0.30, 'medium': 0.50}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PaperGroupStats:
    paper_id: int
    group_name: str
    metric: str
    positive_count: float
    total_count: int
    rate: float


@dataclass
class GroupSummaryStats:
    group_name: str
    metric: str
    total_items: int
    positive_items: float
    n_papers: int
    paper_rates: List[float]
    mean_rate: float
    std_rate: float
    se_rate: float
    ci_lower: float
    ci_upper: float


@dataclass
class PairedComparisonResult:
    group1: str
    group2: str
    metric: str
    n1_items: int
    n2_items: int
    n1_papers: int
    n2_papers: int
    n_paired_papers: int
    mean1: float
    mean2: float
    std1: float
    std2: float
    ci1: Tuple[float, float]
    ci2: Tuple[float, float]
    mean_diff: float
    std_diff: float
    se_diff: float
    ci_diff: Tuple[float, float]
    test_name: str
    test_statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_name: str
    effect_magnitude: str
    direction: str
    classification: str


# ============================================================================
# DATA LOADING
# ============================================================================

# ============================================================================
# GROUP CONSTRUCTION
# ============================================================================

def create_comparison_groups(
    items: List[ReviewItem],
    rankings: Dict[int, Dict[str, str]],
) -> Dict[str, List[ReviewItem]]:
    """See peerreview_analysis.create_comparison_groups for the single-
    human-reviewer special case (items with best == worst are added to
    both Best and Worst Human)."""
    groups = {'Best Human': [], 'Worst Human': [], 'GPT': [], 'Claude': [], 'Gemini': []}
    for item in items:
        if item.reviewer_type == 'Human':
            if item.paper_id in rankings:
                r = rankings[item.paper_id]
                if item.reviewer_id == r.get('best'):
                    groups['Best Human'].append(item)
                # Intentionally not elif — single-reviewer papers have
                # best == worst and the item goes in both groups.
                if item.reviewer_id == r.get('worst'):
                    groups['Worst Human'].append(item)
        elif item.reviewer_type == 'AI':
            if item.model_name in groups:
                groups[item.model_name].append(item)
    return groups


# ============================================================================
# CASCADING FILTERS
# ============================================================================

def get_items_for_correctness(items: List[ReviewItem]) -> List[ReviewItem]:
    return _filter_for_correctness(items)


def get_items_for_significance(items: List[ReviewItem]) -> List[ReviewItem]:
    return _filter_for_significance(items)


def get_items_for_evidence(items: List[ReviewItem]) -> List[ReviewItem]:
    return _filter_for_evidence(items)


# ============================================================================
# PAPER-LEVEL AGGREGATION
# ============================================================================

def calculate_paper_level_stats(items: List[ReviewItem], group_name: str,
                                 metric: str) -> List[PaperGroupStats]:
    if metric == 'correctness':
        filtered = get_items_for_correctness(items)
        get_v = lambda i: i.correctness_numeric
    elif metric == 'significance':
        filtered = get_items_for_significance(items)
        get_v = lambda i: i.significance_numeric
    elif metric == 'evidence':
        filtered = get_items_for_evidence(items)
        get_v = lambda i: i.evidence_numeric
    elif metric == 'fully_good':
        # Fully-good is a property of the full (Correct + Sig + Sufficient)
        # cascade. Denominator is all items in the group (post cascade
        # stripping); numerator is the items that are fully good.
        filtered = list(items)
        get_v = lambda i: 1 if is_fully_good(i) else 0
    else:
        raise ValueError(f'unknown metric: {metric}')

    by_paper: Dict[int, List[ReviewItem]] = {}
    for item in filtered:
        by_paper.setdefault(item.paper_id, []).append(item)

    stats_list = []
    for pid, p_items in by_paper.items():
        total = len(p_items)
        if metric == 'significance':
            vals = [get_v(i) for i in p_items]
            stats_list.append(PaperGroupStats(
                paper_id=pid, group_name=group_name, metric=metric,
                positive_count=float(sum(vals)), total_count=total,
                rate=float(np.mean(vals)) if total > 0 else np.nan,
            ))
        else:
            pos = sum(get_v(i) for i in p_items)
            stats_list.append(PaperGroupStats(
                paper_id=pid, group_name=group_name, metric=metric,
                positive_count=float(pos), total_count=total,
                rate=pos / total if total > 0 else np.nan,
            ))
    return stats_list


def aggregate_paper_stats(paper_stats: List[PaperGroupStats], group_name: str,
                           metric: str) -> GroupSummaryStats:
    valid = [ps for ps in paper_stats if not np.isnan(ps.rate)]
    if not valid:
        return GroupSummaryStats(
            group_name=group_name, metric=metric,
            total_items=0, positive_items=0, n_papers=0, paper_rates=[],
            mean_rate=np.nan, std_rate=np.nan, se_rate=np.nan,
            ci_lower=np.nan, ci_upper=np.nan,
        )
    total_items = sum(ps.total_count for ps in valid)
    positive_items = sum(ps.positive_count for ps in valid)
    rates = [ps.rate for ps in valid]
    n = len(rates)
    mean_r = float(np.mean(rates))
    std_r = float(np.std(rates, ddof=1)) if n > 1 else 0.0
    se_r = std_r / np.sqrt(n) if n > 0 else np.nan
    if n > 1:
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lo, ci_hi = mean_r - t_crit * se_r, mean_r + t_crit * se_r
    else:
        ci_lo = ci_hi = mean_r
    return GroupSummaryStats(
        group_name=group_name, metric=metric,
        total_items=total_items, positive_items=positive_items,
        n_papers=n, paper_rates=rates,
        mean_rate=mean_r, std_rate=std_r, se_rate=se_r,
        ci_lower=ci_lo, ci_upper=ci_hi,
    )


def calculate_all_group_stats(groups: Dict[str, List[ReviewItem]],
                               metric: str) -> Dict[str, GroupSummaryStats]:
    return {name: aggregate_paper_stats(calculate_paper_level_stats(items, name, metric), name, metric)
            for name, items in groups.items()}


# ============================================================================
# EFFECT SIZES
# ============================================================================

def _magnitude_from_thresholds(abs_val: float, thresholds: Dict[str, float]) -> str:
    if abs_val < thresholds['negligible']:
        return 'negligible'
    if abs_val < thresholds['small']:
        return 'small'
    if abs_val < thresholds['medium']:
        return 'medium'
    return 'large'


def cohens_d_paired(differences: np.ndarray) -> Tuple[float, str]:
    n = len(differences)
    if n < 2:
        return np.nan, 'undetermined'
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    if std_diff == 0:
        if mean_diff == 0:
            return 0.0, 'negligible'
        return (np.inf if mean_diff > 0 else -np.inf), 'large'
    d = mean_diff / std_diff
    return d, _magnitude_from_thresholds(abs(d), COHENS_D_THRESHOLDS)


def cohens_d_paired_ci(differences: np.ndarray, confidence: float = 0.95,
                        n_bootstrap: int = 1000) -> Tuple[float, float]:
    n = len(differences)
    if n < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    d_values = []
    for _ in range(n_bootstrap):
        boot = rng.choice(differences, size=n, replace=True)
        d, _ = cohens_d_paired(boot)
        if not np.isnan(d) and not np.isinf(d):
            d_values.append(d)
    if len(d_values) < 10:
        return np.nan, np.nan
    alpha = 1 - confidence
    return (np.percentile(d_values, 100 * alpha / 2),
            np.percentile(d_values, 100 * (1 - alpha / 2)))


def rank_biserial_correlation(differences: np.ndarray) -> Tuple[float, str]:
    nonzero = differences[differences != 0]
    if len(nonzero) < 1:
        return 0.0, 'negligible'
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = np.sum(ranks[nonzero > 0])
    w_minus = np.sum(ranks[nonzero < 0])
    total = w_plus + w_minus
    if total == 0:
        return 0.0, 'negligible'
    r = (w_plus - w_minus) / total
    return r, _magnitude_from_thresholds(abs(r), RANK_BISERIAL_THRESHOLDS)


def _get_effect_size_thresholds(effect_size_name: str) -> Dict[str, float]:
    return RANK_BISERIAL_THRESHOLDS if effect_size_name == 'rank-biserial r' else COHENS_D_THRESHOLDS


# ============================================================================
# PAIRED COMPARISON
# ============================================================================

def compare_groups_paired(group1_name: str, group1_items: List[ReviewItem],
                           group2_name: str, group2_items: List[ReviewItem],
                           metric: str) -> PairedComparisonResult:
    ps1 = calculate_paper_level_stats(group1_items, group1_name, metric)
    ps2 = calculate_paper_level_stats(group2_items, group2_name, metric)
    s1 = aggregate_paper_stats(ps1, group1_name, metric)
    s2 = aggregate_paper_stats(ps2, group2_name, metric)

    rates1 = {p.paper_id: p.rate for p in ps1 if not np.isnan(p.rate)}
    rates2 = {p.paper_id: p.rate for p in ps2 if not np.isnan(p.rate)}
    paired = sorted(set(rates1.keys()) & set(rates2.keys()))
    n_paired = len(paired)

    if n_paired < 2:
        return PairedComparisonResult(
            group1=group1_name, group2=group2_name, metric=metric,
            n1_items=s1.total_items, n2_items=s2.total_items,
            n1_papers=s1.n_papers, n2_papers=s2.n_papers, n_paired_papers=n_paired,
            mean1=s1.mean_rate, mean2=s2.mean_rate,
            std1=s1.std_rate, std2=s2.std_rate,
            ci1=(s1.ci_lower, s1.ci_upper), ci2=(s2.ci_lower, s2.ci_upper),
            mean_diff=np.nan, std_diff=np.nan, se_diff=np.nan,
            ci_diff=(np.nan, np.nan),
            test_name='paired t-test', test_statistic=np.nan,
            p_value=1.0, significant=False,
            effect_size=np.nan, effect_size_name="Cohen's d (paired)",
            effect_magnitude='undetermined',
            direction='undetermined', classification='?',
        )

    differences = np.array([rates1[p] - rates2[p] for p in paired])
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    se_diff = std_diff / np.sqrt(n_paired)
    t_crit = stats.t.ppf(0.975, df=n_paired - 1)
    ci_diff = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

    t_stat, p_value = ttest_1samp(differences, 0)
    d, magnitude = cohens_d_paired(differences)

    return PairedComparisonResult(
        group1=group1_name, group2=group2_name, metric=metric,
        n1_items=s1.total_items, n2_items=s2.total_items,
        n1_papers=s1.n_papers, n2_papers=s2.n_papers, n_paired_papers=n_paired,
        mean1=s1.mean_rate, mean2=s2.mean_rate,
        std1=s1.std_rate, std2=s2.std_rate,
        ci1=(s1.ci_lower, s1.ci_upper), ci2=(s2.ci_lower, s2.ci_upper),
        mean_diff=mean_diff, std_diff=std_diff, se_diff=se_diff, ci_diff=ci_diff,
        test_name='paired t-test', test_statistic=float(t_stat),
        p_value=float(p_value), significant=bool(p_value < ALPHA),
        effect_size=float(d), effect_size_name="Cohen's d (paired)",
        effect_magnitude=magnitude,
        direction='undetermined', classification='?',
    )


def compare_groups_ordinal_paired(group1_name: str, group1_items: List[ReviewItem],
                                   group2_name: str, group2_items: List[ReviewItem],
                                   metric: str = 'significance') -> PairedComparisonResult:
    ps1 = calculate_paper_level_stats(group1_items, group1_name, metric)
    ps2 = calculate_paper_level_stats(group2_items, group2_name, metric)
    s1 = aggregate_paper_stats(ps1, group1_name, metric)
    s2 = aggregate_paper_stats(ps2, group2_name, metric)

    rates1 = {p.paper_id: p.rate for p in ps1 if not np.isnan(p.rate)}
    rates2 = {p.paper_id: p.rate for p in ps2 if not np.isnan(p.rate)}
    paired = sorted(set(rates1.keys()) & set(rates2.keys()))
    n_paired = len(paired)

    if n_paired < 2:
        return PairedComparisonResult(
            group1=group1_name, group2=group2_name, metric=metric,
            n1_items=s1.total_items, n2_items=s2.total_items,
            n1_papers=s1.n_papers, n2_papers=s2.n_papers, n_paired_papers=n_paired,
            mean1=s1.mean_rate, mean2=s2.mean_rate,
            std1=s1.std_rate, std2=s2.std_rate,
            ci1=(s1.ci_lower, s1.ci_upper), ci2=(s2.ci_lower, s2.ci_upper),
            mean_diff=np.nan, std_diff=np.nan, se_diff=np.nan, ci_diff=(np.nan, np.nan),
            test_name='Wilcoxon signed-rank', test_statistic=np.nan,
            p_value=1.0, significant=False,
            effect_size=np.nan, effect_size_name='rank-biserial r',
            effect_magnitude='undetermined',
            direction='undetermined', classification='?',
        )

    paired_vals1 = np.array([rates1[p] for p in paired])
    paired_vals2 = np.array([rates2[p] for p in paired])
    differences = paired_vals1 - paired_vals2

    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    se_diff = std_diff / np.sqrt(n_paired)
    t_crit = stats.t.ppf(0.975, df=n_paired - 1)
    ci_diff = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

    nonzero = differences[differences != 0]
    if len(nonzero) < 1:
        w_stat, p_value = 0.0, 1.0
    else:
        try:
            w_stat, p_value = wilcoxon(paired_vals1, paired_vals2, alternative='two-sided')
        except ValueError:
            w_stat, p_value = 0.0, 1.0

    r, magnitude = rank_biserial_correlation(differences)

    return PairedComparisonResult(
        group1=group1_name, group2=group2_name, metric=metric,
        n1_items=s1.total_items, n2_items=s2.total_items,
        n1_papers=s1.n_papers, n2_papers=s2.n_papers, n_paired_papers=n_paired,
        mean1=s1.mean_rate, mean2=s2.mean_rate,
        std1=s1.std_rate, std2=s2.std_rate,
        ci1=(s1.ci_lower, s1.ci_upper), ci2=(s2.ci_lower, s2.ci_upper),
        mean_diff=mean_diff, std_diff=std_diff, se_diff=se_diff, ci_diff=ci_diff,
        test_name='Wilcoxon signed-rank', test_statistic=float(w_stat),
        p_value=float(p_value), significant=bool(p_value < ALPHA),
        effect_size=float(r), effect_size_name='rank-biserial r',
        effect_magnitude=magnitude,
        direction='undetermined', classification='?',
    )


def _update_classification(r: PairedComparisonResult):
    """Classification with effect size gate using raw p-values."""
    thresholds = _get_effect_size_thresholds(r.effect_size_name)
    if r.significant and not np.isnan(r.effect_size) and abs(r.effect_size) >= thresholds['negligible']:
        if r.mean_diff > 0:
            r.direction = 'group1_better'
            r.classification = '++' if r.effect_magnitude in ('medium', 'large') else '+'
        else:
            r.direction = 'group2_better'
            r.classification = '--' if r.effect_magnitude in ('medium', 'large') else '-'
    else:
        if r.significant:
            r.direction = 'negligible_effect'
            r.classification = '≈'
        else:
            r.direction = 'undetermined'
            r.classification = '?'


def run_all_comparisons(groups: Dict[str, List[ReviewItem]], metric: str) -> List[PairedComparisonResult]:
    group_names = ['Best Human', 'Worst Human', 'GPT', 'Claude', 'Gemini']
    # Binary rate: paired t-test + Cohen's d (used for correctness, evidence, fully_good)
    # Ordinal: Wilcoxon signed-rank + rank-biserial r (used for significance)
    compare_func = compare_groups_ordinal_paired if metric == 'significance' else compare_groups_paired
    results = []
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i+1:]:
            if g1 not in groups or g2 not in groups:
                continue
            if not groups[g1] or not groups[g2]:
                continue
            results.append(compare_func(g1, groups[g1], g2, groups[g2], metric))
    # Classify using raw p-values + effect size gate
    for r in results:
        _update_classification(r)
    return results


# ============================================================================
# REPORTING
# ============================================================================

def _item_counts(group_items: List[ReviewItem], metric: str) -> Dict:
    """Return the raw positive/negative/distribution counts for Table N columns."""
    if metric == 'correctness':
        filtered = _filter_for_correctness(group_items)
        pos = sum(1 for i in filtered if i.correctness_numeric == 1)
        neg = sum(1 for i in filtered if i.correctness_numeric == 0)
        return {'n_items': len(filtered), 'positive': pos, 'negative': neg}
    elif metric == 'significance':
        filtered = _filter_for_significance(group_items)
        not_sig = sum(1 for i in filtered if i.significance_numeric == 0)
        marg = sum(1 for i in filtered if i.significance_numeric == 1)
        sig = sum(1 for i in filtered if i.significance_numeric == 2)
        return {'n_items': len(filtered), 'not_sig': not_sig, 'marg': marg, 'sig': sig}
    else:
        filtered = _filter_for_evidence(group_items)
        pos = sum(1 for i in filtered if i.evidence_numeric == 1)
        neg = sum(1 for i in filtered if i.evidence_numeric == 0)
        return {'n_items': len(filtered), 'positive': pos, 'negative': neg}


def generate_descriptive_table(groups: Dict[str, List[ReviewItem]], metric: str) -> str:
    lines = []
    group_order = ['Best Human', 'Worst Human', 'GPT', 'Claude', 'Gemini']
    stats_dict = calculate_all_group_stats(groups, metric)

    if metric == 'correctness':
        title = "TABLE 1: CORRECTNESS RATE (per-paper mean, paper-level)"
    elif metric == 'significance':
        title = "TABLE 2: SIGNIFICANCE SCORE (per-paper mean on 0-2 scale, paper-level)"
    else:
        title = "TABLE 3: EVIDENCE SUFFICIENCY RATE (per-paper mean, paper-level)"
    lines.append(title)
    lines.append("-" * 120)

    if metric == 'correctness':
        lines.append(
            f"{'Group':<15} {'Mean Rate':>10} {'95% CI':>18} {'SD':>8} "
            f"{'Correct':>9} {'Incorrect':>11} {'Total Items':>13} {'(# papers)':>12}"
        )
    elif metric == 'significance':
        lines.append(
            f"{'Group':<15} {'Mean':>8} {'95% CI':>18} {'SD':>8} "
            f"{'Not Sig.':>9} {'Marg.':>7} {'Sig.':>7} {'Total Items':>13} {'(# papers)':>12}"
        )
    else:
        lines.append(
            f"{'Group':<15} {'Mean Rate':>10} {'95% CI':>18} {'SD':>8} "
            f"{'Sufficient':>11} {'Not Suff.':>11} {'Total Items':>13} {'(# papers)':>12}"
        )
    lines.append("-" * 120)

    for g in group_order:
        if g not in stats_dict:
            continue
        s = stats_dict[g]
        counts = _item_counts(groups[g], metric)
        if np.isnan(s.mean_rate):
            if metric == 'correctness':
                lines.append(
                    f"{g:<15} {'N/A':>10} {'N/A':>18} {'N/A':>8} "
                    f"{0:>9} {0:>11} {0:>13} {'(0)':>12}"
                )
            elif metric == 'significance':
                lines.append(
                    f"{g:<15} {'N/A':>8} {'N/A':>18} {'N/A':>8} "
                    f"{0:>9} {0:>7} {0:>7} {0:>13} {'(0)':>12}"
                )
            else:
                lines.append(
                    f"{g:<15} {'N/A':>10} {'N/A':>18} {'N/A':>8} "
                    f"{0:>11} {0:>11} {0:>13} {'(0)':>12}"
                )
            continue
        papers_tag = f"({s.n_papers})"
        if metric == 'correctness':
            ci = f"[{s.ci_lower:.1%}, {s.ci_upper:.1%}]"
            lines.append(
                f"{g:<15} {s.mean_rate:>10.1%} {ci:>18} {s.std_rate:>8.1%} "
                f"{counts['positive']:>9} {counts['negative']:>11} "
                f"{s.total_items:>13} {papers_tag:>12}"
            )
        elif metric == 'significance':
            ci = f"[{s.ci_lower:.2f}, {s.ci_upper:.2f}]"
            lines.append(
                f"{g:<15} {s.mean_rate:>8.2f} {ci:>18} {s.std_rate:>8.2f} "
                f"{counts['not_sig']:>9} {counts['marg']:>7} {counts['sig']:>7} "
                f"{s.total_items:>13} {papers_tag:>12}"
            )
        else:
            ci = f"[{s.ci_lower:.1%}, {s.ci_upper:.1%}]"
            lines.append(
                f"{g:<15} {s.mean_rate:>10.1%} {ci:>18} {s.std_rate:>8.1%} "
                f"{counts['positive']:>11} {counts['negative']:>11} "
                f"{s.total_items:>13} {papers_tag:>12}"
            )
    lines.append("")
    return "\n".join(lines)


def generate_comparison_table(results: List[PairedComparisonResult], metric_name: str) -> str:
    """Pairwise comparison table (replaces the 5×5 matrix).

    Ordinal handling is keyed on the metric_name — only 'significance' is
    treated as ordinal. Correctness, Evidence, and Fully Good are all binary
    rates and use the same Cohen's d / paired t-test framing."""
    is_ordinal = (metric_name.lower() == 'significance')
    effect_label = 'r (rank-biserial)' if is_ordinal else "d (Cohen's d)"
    lines = []
    lines.append(f"\nPAIRWISE COMPARISONS — {metric_name.upper()}")
    lines.append("-" * 130)
    lines.append(
        f"{'Comparison':<28} {'Mean1':>8} {'Mean2':>8} {'Diff':>9} {'95% CI (diff)':>22} "
        f"{'p-value':>12} {effect_label:>20} {'magnitude':>12} {'N(paired)':>10}"
    )
    lines.append("-" * 130)
    for r in sorted(results, key=lambda x: (np.inf if np.isnan(x.p_value) else x.p_value)):
        cmp = f"{r.group1} vs {r.group2}"
        if is_ordinal:
            m1 = f"{r.mean1:.2f}" if not np.isnan(r.mean1) else "N/A"
            m2 = f"{r.mean2:.2f}" if not np.isnan(r.mean2) else "N/A"
            if not np.isnan(r.mean_diff):
                diff_s = f"{r.mean_diff:+.2f}"
                ci_s = f"[{r.ci_diff[0]:+.2f}, {r.ci_diff[1]:+.2f}]"
            else:
                diff_s, ci_s = "N/A", "N/A"
        else:
            m1 = f"{r.mean1:.1%}" if not np.isnan(r.mean1) else "N/A"
            m2 = f"{r.mean2:.1%}" if not np.isnan(r.mean2) else "N/A"
            if not np.isnan(r.mean_diff):
                diff_s = f"{r.mean_diff:+.1%}"
                ci_s = f"[{r.ci_diff[0]:+.1%}, {r.ci_diff[1]:+.1%}]"
            else:
                diff_s, ci_s = "N/A", "N/A"
        p_s = f"{r.p_value:.4f}" if not np.isnan(r.p_value) else "N/A"
        if r.significant:
            p_s += " *"
        es = f"{r.effect_size:+.3f}" if not np.isnan(r.effect_size) else "N/A"
        lines.append(
            f"{cmp:<28} {m1:>8} {m2:>8} {diff_s:>9} {ci_s:>22} {p_s:>12} "
            f"{es:>20} {r.effect_magnitude:>12} {r.n_paired_papers:>10}"
        )
    lines.append("")
    lines.append(f"  * = p < {ALPHA}")
    lines.append("")
    return "\n".join(lines)


def generate_full_report(groups: Dict[str, List[ReviewItem]],
                          correctness_results: List[PairedComparisonResult],
                          significance_results: List[PairedComparisonResult],
                          evidence_results: List[PairedComparisonResult],
                          fully_good_results: List[PairedComparisonResult]) -> str:
    lines = []
    lines.append("=" * 110)
    lines.append("PEERREVIEW BENCH — PER-PAPER PAIRED ANALYSIS REPORT")
    lines.append("=" * 110)
    lines.append("")
    lines.append("METHODOLOGY")
    lines.append("-" * 110)
    lines.append("")
    lines.append("  Data source: the `expert_annotation` config of the `prometheus-eval/peerreview-bench`")
    lines.append("  HuggingFace dataset. Both primary and secondary annotators' rows are included as")
    lines.append("  independent data points (merged semantics).")
    lines.append("")
    lines.append("  Framing: for each paper, compute a per-group mean rate (binary metrics) or")
    lines.append("  per-group mean score (ordinal significance). Pair the per-paper means across")
    lines.append("  groups on the papers where BOTH groups have data, then run the paired test.")
    lines.append("")
    lines.append("  Per-metric test, effect size, and confidence interval:")
    lines.append("")
    lines.append("    Correctness (binary: Correct / Not Correct)")
    lines.append("      - Item-level rate CI:   Wilson score interval")
    lines.append("      - Paired paper-level:   paired t-test on per-paper rate differences")
    lines.append("      - Effect size:          Cohen's d (paired) = mean(diff) / SD(diff)")
    lines.append("      - Diff CI:              t-interval on the paired paper-level differences")
    lines.append("")
    lines.append("    Significance (ordinal 3-class: Not Sig / Marginally Sig / Significant, encoded 0/1/2)")
    lines.append("      - Item-level mean CI:   t-interval on per-paper means")
    lines.append("      - Paired paper-level:   Wilcoxon signed-rank on per-paper mean differences")
    lines.append("      - Effect size:          rank-biserial r (Kerby 2014)")
    lines.append("      - Diff CI:              t-interval on the paired paper-level differences")
    lines.append("")
    lines.append("    Evidence (binary: Sufficient / Requires More)")
    lines.append("      - Item-level rate CI:   Wilson score interval")
    lines.append("      - Paired paper-level:   paired t-test on per-paper rate differences")
    lines.append("      - Effect size:          Cohen's d (paired) = mean(diff) / SD(diff)")
    lines.append("      - Diff CI:              t-interval on the paired paper-level differences")
    lines.append("")
    lines.append(f"  Significance level: α = {ALPHA}. Raw p-values are reported.")
    lines.append("")

    lines.append("SAMPLE SIZES (per reviewer group)")
    lines.append("-" * 110)
    for name, items in groups.items():
        n_papers = len(set(i.paper_id for i in items))
        lines.append(f"  {name:15} n_items={len(items):4}  n_papers={n_papers:3}")
    lines.append("")

    lines.append("=" * 110)
    lines.append("DESCRIPTIVE STATISTICS (PAPER-LEVEL)")
    lines.append("=" * 110)
    lines.append("")
    for metric in ['correctness', 'significance', 'evidence']:
        lines.append(generate_descriptive_table(groups, metric))

    lines.append("")
    lines.append("=" * 110)
    lines.append("PAIRWISE PAIRED COMPARISONS")
    lines.append("=" * 110)
    for metric_name, results in [('Correctness', correctness_results),
                                  ('Significance', significance_results),
                                  ('Evidence', evidence_results),
                                  ('Fully Good', fully_good_results)]:
        lines.append(generate_comparison_table(results, metric_name))

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def run_analysis(items: List[ReviewItem], rankings: Dict[int, Dict[str, str]],
                 output_dir: Optional[str] = None) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("PEERREVIEW BENCH — PER-PAPER PAIRED ANALYSIS")
    print("=" * 60)

    # --- Dataset overview ---
    print("\n1. Computing dataset overview...")
    overview = summarize_dataset(items)
    overview_report = format_dataset_overview(overview)
    print(overview_report)

    kept_items = filter_kept_items(items)

    print("\n2. Creating comparison groups...")
    groups = create_comparison_groups(kept_items, rankings)
    for name, g_items in groups.items():
        n_papers = len(set(i.paper_id for i in g_items))
        print(f"   {name:15} n_items={len(g_items):4}, n_papers={n_papers:3}")

    print("\n3. Computing fully-good stats per group...")
    fully_good = compute_fully_good_stats(groups)
    fully_good_report = format_fully_good_report(fully_good)
    print(fully_good_report)

    print("\n4. Running comparisons...")
    correctness_results = run_all_comparisons(groups, 'correctness')
    significance_results = run_all_comparisons(groups, 'significance')
    evidence_results = run_all_comparisons(groups, 'evidence')
    fully_good_results = run_all_comparisons(groups, 'fully_good')

    print("\n5. Generating report...")
    report = (overview_report + "\n" + fully_good_report + "\n" +
              generate_full_report(groups, correctness_results, significance_results,
                                    evidence_results, fully_good_results))
    print(report)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'analysis_report_per_paper.txt', 'w') as f:
            f.write(report)
        import json as _json
        with open(out / 'dataset_overview.json', 'w') as f:
            _json.dump(overview, f, indent=2, default=str)
        with open(out / 'fully_good_stats.json', 'w') as f:
            _json.dump(fully_good, f, indent=2)
        for name, results in [('correctness', correctness_results),
                              ('significance', significance_results),
                              ('evidence', evidence_results),
                              ('fully_good', fully_good_results)]:
            rows = [{
                'group1': r.group1, 'group2': r.group2, 'metric': r.metric,
                'mean1': r.mean1, 'mean2': r.mean2,
                'mean_diff': r.mean_diff, 'std_diff': r.std_diff,
                'ci_diff_lower': r.ci_diff[0], 'ci_diff_upper': r.ci_diff[1],
                'n_paired_papers': r.n_paired_papers,
                'test_statistic': r.test_statistic, 'p_value': r.p_value,
                'effect_size': r.effect_size, 'effect_magnitude': r.effect_magnitude,
            } for r in results]
            pd.DataFrame(rows).to_csv(out / f'{name}_comparisons_per_paper.csv', index=False)
        print(f"\n6. Results saved to: {output_dir}")

    return {
        'groups': {k: len(v) for k, v in groups.items()},
        'correctness_results': correctness_results,
        'significance_results': significance_results,
        'evidence_results': evidence_results,
        'report': report,
    }


def main():
    parser = argparse.ArgumentParser(description='PeerReview Bench Per-Paper Analysis')
    parser.add_argument('--output-dir', type=str, default='./analysis_output_per_paper')
    parser.add_argument('--annotator-source', type=str, default='both',
                        choices=['primary', 'secondary', 'both'],
                        help="Default 'both' merges primary and secondary as "
                             "independent data points.")
    args = parser.parse_args()

    print(f"Loading annotations (annotator_source={args.annotator_source}) from HuggingFace dataset...")
    items, rankings = load_annotations(annotator_source=args.annotator_source)
    print(f"Loaded {len(items)} items, {len(rankings)} paper rankings")

    run_analysis(items, rankings, args.output_dir)
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
