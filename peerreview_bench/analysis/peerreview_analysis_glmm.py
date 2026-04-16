#!/usr/bin/env python3
"""
PeerReview Bench - GLMM Robustness Analysis

Fits generalized linear mixed models as a sensitivity check for the primary
paired paper-level analysis.

Models:
  - Correctness (binary):  True logistic GLMM via BinomialBayesMixedGLM
                           (variational Bayes, logit link, paper random intercept).
                           Falls back to GEE with Binomial family + exchangeable
                           correlation if the Bayes fit fails to converge.
  - Evidence    (binary):  Same as Correctness.
  - Significance (ordinal 0/1/2): Two binomial GLMMs at cut points:
      (a) P(Y ≥ 1) — "at least marginally significant"
      (b) P(Y = 2) — "significant"
    plus a linear mixed model on the raw 0-2 scores as an additional sensitivity check
    (treating ordinal as continuous, acknowledged approximation).

For logistic GLMM, ICC is computed on the logit scale:
    ICC = σ²_u / (σ²_u + π²/3),  where π²/3 ≈ 3.2899

Usage:
    python peerreview_analysis_glmm.py --combined <combined.json> --rankings-file <rankings.json> --output-dir <output_dir>
"""

import json
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

# Allow `load_data` imports from the parent peerreview_bench/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from data_filter import (
    CATEGORIES_FOR_CORRECTNESS, CATEGORIES_FOR_SIGNIFICANCE, CATEGORIES_FOR_EVIDENCE,
    FULLY_DROPPED_CATEGORIES,
    classify_item_validity,
    N_TOTAL_ANNOTATORS, N_PRIMARY_ANNOTATORS, N_SECONDARY_ANNOTATORS,
    N_ANNOTATORS_BOTH_ROLES,
)
from load_data import load_dataframe

warnings.filterwarnings('ignore')

try:
    import statsmodels.formula.api as smf
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not found. Install with: pip install statsmodels")


PI_SQ_OVER_3 = (np.pi ** 2) / 3  # ~3.2899, variance of standard logistic distribution


# ============================================================================
# DATA LOADING
# ============================================================================

# Note: data loading is delegated to load_data.load_dataframe()


def _classify_df_row(row: pd.Series) -> str:
    """Classify a GLMM DataFrame row using the shared data_filter rules.

    The GLMM DataFrame uses short column names ('correctness', 'significance',
    'evidence') while classify_item_validity expects the '_numeric' suffix form;
    we adapt here.
    """
    def _v(x):
        return None if pd.isna(x) else int(x)
    return classify_item_validity({
        'correctness_numeric': _v(row['correctness']),
        'significance_numeric': _v(row['significance']),
        'evidence_numeric': _v(row['evidence']),
    })


def apply_data_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-dropped categories, add a 'category' column."""
    df = df.copy()
    df['category'] = df.apply(_classify_df_row, axis=1)
    df = df[~df['category'].isin(FULLY_DROPPED_CATEGORIES)].copy()
    return df


def assign_groups(df: pd.DataFrame, rankings: Dict) -> pd.DataFrame:
    """Assign each item to Best Human, Worst Human, GPT, Claude, or Gemini.

    On single-human-reviewer papers (best == worst), the one human's items
    are emitted as duplicate rows, one per group. This keeps Best and Worst
    Human at identical paper coverage. The GLMM treats the duplicates as
    independent observations, which is a slight over-count; the bias is
    negligible given how few such papers there are (~2% of the dataset)
    and each duplicated row contributes zero paired difference to the
    Best vs Worst contrast at the paper-random-effect level.
    """
    def get_groups_for_row(row):
        """Return a list of groups this row belongs to (1 or 2 elements)."""
        if row['reviewer_type'] == 'AI':
            model = row['model_name']
            return [model] if model in ('GPT', 'Claude', 'Gemini') else []
        if row['reviewer_type'] != 'Human':
            return []
        pid = row['paper_id']
        if pid not in rankings:
            return []
        r = rankings[pid]
        groups_out = []
        if row['reviewer_id'] == r.get('best'):
            groups_out.append('Best Human')
        if row['reviewer_id'] == r.get('worst'):
            groups_out.append('Worst Human')
        return groups_out

    df = df.copy()
    df['_groups'] = df.apply(get_groups_for_row, axis=1)
    df = df[df['_groups'].map(len) > 0].copy()
    # Explode multi-group rows into one row per group
    df = df.explode('_groups').rename(columns={'_groups': 'group'})
    return df.copy()


def _prepare_data(df: pd.DataFrame, outcome_col: str, group_col: str,
                   cluster_col: str, reference_group: str) -> pd.DataFrame:
    data = df[df[outcome_col].notna()].copy()
    data[outcome_col] = data[outcome_col].astype(int)
    data[cluster_col] = data[cluster_col].astype(str)
    other_groups = sorted([g for g in data[group_col].unique() if g != reference_group])
    data[group_col] = pd.Categorical(
        data[group_col],
        categories=[reference_group] + other_groups,
    )
    return data


# ============================================================================
# TRUE LOGISTIC GLMM via BinomialBayesMixedGLM
# ============================================================================

def _extract_bayes_fe_summary(fit) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Extract fixed-effect names, posterior means, and posterior SDs from a BayesMixedGLMResults."""
    fe_names = list(fit.model.exog_names)
    n_fe = len(fe_names)
    fe_mean = np.asarray(fit.fe_mean)[:n_fe]
    fe_sd = np.asarray(fit.fe_sd)[:n_fe]
    z = fe_mean / np.where(fe_sd > 0, fe_sd, np.nan)
    return fe_names, fe_mean, fe_sd, z


def _compute_bayes_icc(fit) -> float:
    """
    For BinomialBayesMixedGLM, vcp_mean is the posterior mean of the log
    standard deviation of the random effect. So σ_u = exp(vcp_mean)
    and the logit-scale ICC = σ²_u / (σ²_u + π²/3).
    """
    try:
        vcp_mean = float(np.asarray(fit.vcp_mean).flat[0])
        sigma_u = np.exp(vcp_mean)
        var_u = sigma_u ** 2
        if not np.isfinite(var_u) or var_u <= 0:
            return np.nan
        return var_u / (var_u + PI_SQ_OVER_3)
    except Exception:
        return np.nan


def fit_logistic_glmm(df: pd.DataFrame, outcome_col: str,
                       group_col: str = 'group', cluster_col: str = 'paper_id',
                       reference_group: str = 'Best Human') -> Dict:
    """Fit a true logistic GLMM. Primary: BinomialBayesMixedGLM. Fallback: GEE-Binomial."""
    data = _prepare_data(df, outcome_col, group_col, cluster_col, reference_group)

    n_items = len(data)
    n_papers = data[cluster_col].nunique()
    n_groups = data[group_col].nunique()

    print(f"\n  Fitting logistic GLMM for {outcome_col}")
    print(f"    N items: {n_items}, N papers: {n_papers}, N groups: {n_groups}")
    print(f"    Reference group: {reference_group}")

    results = {
        'outcome': outcome_col,
        'reference': reference_group,
        'n_items': int(n_items),
        'n_papers': int(n_papers),
    }

    formula = f"{outcome_col} ~ C({group_col}, Treatment(reference='{reference_group}'))"

    # --- Primary: BinomialBayesMixedGLM ---
    try:
        vc_formulas = {cluster_col: f"0 + C({cluster_col})"}
        model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, data)
        fit = model.fit_vb()

        fe_names, fe_mean, fe_sd, z = _extract_bayes_fe_summary(fit)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z)))
        ci_lower = fe_mean - 1.96 * fe_sd
        ci_upper = fe_mean + 1.96 * fe_sd

        icc = _compute_bayes_icc(fit)
        sigma_u = np.exp(float(np.asarray(fit.vcp_mean).flat[0]))

        results['method'] = 'BinomialBayesMixedGLM (variational Bayes, logit link)'
        results['converged'] = True
        results['fixed_effects'] = dict(zip(fe_names, fe_mean.tolist()))
        results['std_errors'] = dict(zip(fe_names, fe_sd.tolist()))
        results['pvalues'] = dict(zip(fe_names, p_values.tolist()))
        results['conf_int_lower'] = dict(zip(fe_names, ci_lower.tolist()))
        results['conf_int_upper'] = dict(zip(fe_names, ci_upper.tolist()))
        results['icc'] = float(icc) if not np.isnan(icc) else np.nan
        results['re_variance'] = float(sigma_u ** 2)
        results['re_sd'] = float(sigma_u)

        try:
            results['summary'] = str(fit.summary())
        except Exception:
            results['summary'] = f"BinomialBayesMixedGLM fit summary unavailable; fe={dict(zip(fe_names, fe_mean.tolist()))}"

        print(f"    Converged (Bayes VB)")
        if not np.isnan(icc):
            print(f"    ICC (logit scale): {icc:.4f}")
        return results

    except Exception as e1:
        print(f"    BinomialBayesMixedGLM failed: {e1}")
        print(f"    Falling back to GEE (Binomial, exchangeable)...")

    # --- Fallback: GEE with Binomial ---
    try:
        model = GEE.from_formula(formula, groups=cluster_col, data=data,
                                  family=Binomial(), cov_struct=Exchangeable())
        fit = model.fit()

        fe_names = list(fit.params.index)
        fe_mean = fit.params.values
        fe_sd = fit.bse.values
        p_values = fit.pvalues.values
        ci_df = fit.conf_int()
        ci_lower = ci_df[0].values
        ci_upper = ci_df[1].values

        dep = fit.cov_struct.dep_params
        icc = float(dep) if np.isscalar(dep) else float(np.asarray(dep).flat[0])

        results['method'] = 'GEE (Binomial, exchangeable correlation) — GLMM fallback'
        results['converged'] = True
        results['fixed_effects'] = dict(zip(fe_names, fe_mean.tolist()))
        results['std_errors'] = dict(zip(fe_names, fe_sd.tolist()))
        results['pvalues'] = dict(zip(fe_names, p_values.tolist()))
        results['conf_int_lower'] = dict(zip(fe_names, ci_lower.tolist()))
        results['conf_int_upper'] = dict(zip(fe_names, ci_upper.tolist()))
        results['icc'] = icc
        results['icc_note'] = 'exchangeable working correlation (not a true GLMM ICC)'
        results['summary'] = str(fit.summary())

        print(f"    GEE converged")
        print(f"    Exchangeable correlation (GEE): {icc:.4f}")
        return results

    except Exception as e2:
        print(f"    GEE also failed: {e2}")
        results['method'] = 'FAILED'
        results['error'] = str(e2)
        return results


# ============================================================================
# ORDINAL SIGNIFICANCE — two-threshold binomial GLMMs + LMM sensitivity
# ============================================================================

def fit_ordinal_cumulative(df: pd.DataFrame, group_col: str = 'group',
                            cluster_col: str = 'paper_id',
                            reference_group: str = 'Best Human') -> Dict:
    """
    For 3-level ordinal significance (0,1,2) we fit two binomial GLMMs:
      - At_Least_Marginal:  P(Y >= 1)
      - Significant:        P(Y == 2)

    This is a non-proportional-odds relaxation of ordinal regression and
    avoids treating the ordinal as continuous.
    """
    data_base = df[(df['correctness'] == 1) & (df['significance'].notna())].copy()
    if len(data_base) < 2:
        return {'method': 'FAILED', 'error': 'insufficient data', 'n_items': 0, 'n_papers': 0}

    out = {
        'outcome': 'significance (cumulative)',
        'reference': reference_group,
        'n_items': int(len(data_base)),
        'n_papers': int(data_base[cluster_col].nunique()),
        'method': 'Two binomial GLMMs at cut points (non-proportional odds)',
        'cutpoints': {},
    }

    for label, threshold in (('at_least_marginal', 1), ('significant', 2)):
        tmp = data_base.copy()
        tmp['y'] = (tmp['significance'] >= threshold).astype(int)
        res = fit_logistic_glmm(tmp, 'y',
                                 group_col=group_col, cluster_col=cluster_col,
                                 reference_group=reference_group)
        out['cutpoints'][label] = res

    return out


def fit_ordinal_lmm(df: pd.DataFrame, group_col: str = 'group',
                     cluster_col: str = 'paper_id',
                     reference_group: str = 'Best Human') -> Dict:
    """Linear mixed model treating significance (0-2) as continuous — sensitivity check."""
    data = df[(df['correctness'] == 1) & (df['significance'].notna())].copy()
    data['significance'] = data['significance'].astype(int)
    data[cluster_col] = data[cluster_col].astype(str)
    other = sorted([g for g in data[group_col].unique() if g != reference_group])
    data[group_col] = pd.Categorical(data[group_col], categories=[reference_group] + other)

    n_items = len(data)
    n_papers = data[cluster_col].nunique()

    print(f"\n  Fitting LMM for significance (ordinal 0-2 as continuous)")
    print(f"    N items: {n_items}, N papers: {n_papers}")

    results = {
        'outcome': 'significance',
        'reference': reference_group,
        'n_items': int(n_items),
        'n_papers': int(n_papers),
    }

    try:
        formula = f"significance ~ C({group_col}, Treatment(reference='{reference_group}'))"
        model = smf.mixedlm(formula, data, groups=data[cluster_col])
        fit = model.fit(reml=False, method='lbfgs', maxiter=500)

        results['method'] = 'Linear Mixed Model (ordinal as continuous — sensitivity)'
        results['converged'] = fit.converged
        results['aic'] = float(fit.aic)
        results['bic'] = float(fit.bic)
        results['fixed_effects'] = {k: float(v) for k, v in fit.params.items()}
        results['std_errors'] = {k: float(v) for k, v in fit.bse.items()}
        results['pvalues'] = {k: float(v) for k, v in fit.pvalues.items()}
        ci_df = fit.conf_int()
        results['conf_int_lower'] = {k: float(ci_df.loc[k, 0]) for k in ci_df.index}
        results['conf_int_upper'] = {k: float(ci_df.loc[k, 1]) for k in ci_df.index}

        re_var = fit.cov_re.iloc[0, 0] if hasattr(fit.cov_re, 'iloc') else float(fit.cov_re)
        resid_var = fit.scale
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0
        results['icc'] = float(icc)
        results['re_variance'] = float(re_var)
        results['residual_variance'] = float(resid_var)
        results['summary'] = str(fit.summary())

        print(f"    Converged: {fit.converged}")
        print(f"    ICC (LMM, normal scale): {icc:.4f}")

    except Exception as e:
        print(f"    LMM failed: {e}")
        results['method'] = 'FAILED'
        results['error'] = str(e)

    return results


# ============================================================================
# REPORTING
# ============================================================================

def _format_param_name(name: str) -> str:
    display = name
    for needle in ("C(group, Treatment(reference='Best Human'))", "C(group, Treatment(reference='Worst Human'))"):
        display = display.replace(needle, '')
    display = display.replace('[T.', '').replace(']', '')
    return display if display else name


def format_results(results: Dict, section_title: str) -> str:
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  {section_title.upper()}")
    lines.append(f"{'=' * 80}")

    if results.get('method') == 'FAILED':
        lines.append(f"  Model fitting failed: {results.get('error', 'unknown')}")
        return "\n".join(lines)

    lines.append(f"  Method:          {results['method']}")
    lines.append(f"  N items:         {results.get('n_items', 'N/A')}")
    lines.append(f"  N papers:        {results.get('n_papers', 'N/A')}")
    lines.append(f"  Reference group: {results.get('reference', 'N/A')}")
    if 'converged' in results:
        lines.append(f"  Converged:       {results['converged']}")
    if 'icc' in results and not (isinstance(results['icc'], float) and np.isnan(results['icc'])):
        note = f"  ({results['icc_note']})" if 'icc_note' in results else ""
        lines.append(f"  ICC:             {results['icc']:.4f}{note}")
        lines.append(f"                   -> {results['icc'] * 100:.1f}% between-paper variance")
    if 'aic' in results:
        lines.append(f"  AIC / BIC:       {results['aic']:.1f} / {results['bic']:.1f}")

    lines.append(f"\n  Fixed Effects (ref = {results.get('reference', 'N/A')}):")
    lines.append(f"  {'Parameter':<45} {'Coef':>10} {'SE':>10} {'p':>10} {'95% CI':>24}")
    lines.append(f"  {'-' * 101}")

    fe = results.get('fixed_effects', {})
    se = results.get('std_errors', {})
    pv = results.get('pvalues', {})
    cil = results.get('conf_int_lower', {})
    cih = results.get('conf_int_upper', {})

    for param, coef in fe.items():
        std_err = se.get(param, np.nan)
        p_val = pv.get(param, np.nan)
        lo = cil.get(param, np.nan)
        hi = cih.get(param, np.nan)
        marker = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
        p_str = f"{p_val:.4f}{marker}" if not np.isnan(p_val) else 'N/A'
        ci_str = f"[{lo:.4f}, {hi:.4f}]" if not (np.isnan(lo) or np.isnan(hi)) else 'N/A'
        lines.append(f"  {_format_param_name(param):<45} {coef:>10.4f} {std_err:>10.4f} {p_str:>10} {ci_str:>24}")
    lines.append("")
    return "\n".join(lines)


def format_cumulative_results(results: Dict, header: str) -> str:
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  {header.upper()}")
    lines.append(f"{'=' * 80}")
    if results.get('method') == 'FAILED':
        lines.append(f"  {results.get('error', 'FAILED')}")
        return "\n".join(lines)
    lines.append(f"  Method: {results['method']}")
    lines.append(f"  N items: {results['n_items']}, N papers: {results['n_papers']}")
    for label, sub in results.get('cutpoints', {}).items():
        pretty = {'at_least_marginal': 'P(Y ≥ 1) — at least marginally significant',
                  'significant': 'P(Y = 2) — significant'}[label]
        lines.append(format_results(sub, pretty))
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PeerReview Bench GLMM Analysis')
    parser.add_argument('--output-dir', type=str, default='./glmm_output')
    parser.add_argument('--annotator-source', type=str, default='both',
                        choices=['primary', 'secondary', 'both'],
                        help="Default 'both' merges primary and secondary as "
                             "independent data points.")
    args = parser.parse_args()

    if not HAS_STATSMODELS:
        print("ERROR: statsmodels is required. Install: pip install statsmodels")
        return

    print(f"Loading annotations (annotator_source={args.annotator_source}) from HuggingFace dataset...")
    df, rankings = load_dataframe(annotator_source=args.annotator_source)
    print(f"  Loaded {len(df)} raw items, {len(rankings)} paper rankings")

    # --- Dataset overview (no rule mentions) ---
    n_papers_raw = int(df['paper_id'].nunique())
    # df uses numeric-label column names ('correctness', 'significance', 'evidence')
    # because it comes from load_dataframe. Convert rows to the ReviewItem-like
    # dict shape that summarize_dataset expects.
    rows_as_items = df.rename(columns={
        'correctness': 'correctness_numeric',
        'significance': 'significance_numeric',
        'evidence': 'evidence_numeric',
    })[['paper_id', 'reviewer_id', 'correctness_numeric',
        'significance_numeric', 'evidence_numeric']].to_dict('records')
    from data_filter import summarize_dataset, format_dataset_overview
    overview = summarize_dataset(rows_as_items)
    overview_report = format_dataset_overview(overview)
    print(overview_report)

    # Apply cascade stripping
    df = apply_data_filter(df)
    print(f"  Items kept for analysis: {len(df)}")

    df = assign_groups(df, rankings)
    print(f"  Items with group assignment: {len(df)}")
    print(f"  Groups: {df['group'].value_counts().to_dict()}")

    # Metric-specific DataFrames (using the category column set by apply_data_filter)
    df_corr = df[df['category'].isin(CATEGORIES_FOR_CORRECTNESS)].copy()
    df_sig = df[df['category'].isin(CATEGORIES_FOR_SIGNIFICANCE)].copy()
    df_evi = df[df['category'].isin(CATEGORIES_FOR_EVIDENCE)].copy()
    print(f"  Per-metric N: correctness={len(df_corr)}, significance={len(df_sig)}, evidence={len(df_evi)}")

    print("\n" + "=" * 80)
    print("GLMM ROBUSTNESS ANALYSIS")
    print("=" * 80)

    all_results = {}
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PEERREVIEW BENCH - GLMM ROBUSTNESS ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Purpose: Sensitivity check for the primary paired paper-level analysis.")
    report_lines.append("")
    report_lines.append("Primary model for binary outcomes: True logistic GLMM via")
    report_lines.append("  BinomialBayesMixedGLM (variational Bayes, logit link, paper random intercept).")
    report_lines.append("  Falls back to GEE (Binomial, exchangeable) if VB fails.")
    report_lines.append("")
    report_lines.append("For ordinal significance (0,1,2): Two binomial GLMMs at cut points")
    report_lines.append("  (P(Y ≥ 1) and P(Y = 2)), plus a linear mixed model sensitivity check.")
    report_lines.append("")
    report_lines.append("ICC on logit scale: σ²_u / (σ²_u + π²/3).")
    report_lines.append("")

    # Correctness (binary)
    corr_best = fit_logistic_glmm(df_corr, 'correctness', reference_group='Best Human')
    corr_worst = fit_logistic_glmm(df_corr, 'correctness', reference_group='Worst Human')
    all_results['correctness_vs_best'] = corr_best
    all_results['correctness_vs_worst'] = corr_worst
    report_lines.append(format_results(corr_best, 'Correctness (ref: Best Human)'))
    report_lines.append(format_results(corr_worst, 'Correctness (ref: Worst Human)'))

    # Significance: cumulative binomial GLMMs (uses significance-eligible items)
    sig_cum_best = fit_ordinal_cumulative(df_sig, reference_group='Best Human')
    sig_cum_worst = fit_ordinal_cumulative(df_sig, reference_group='Worst Human')
    all_results['significance_cumulative_vs_best'] = sig_cum_best
    all_results['significance_cumulative_vs_worst'] = sig_cum_worst
    report_lines.append(format_cumulative_results(sig_cum_best, 'Significance cumulative (ref: Best Human)'))
    report_lines.append(format_cumulative_results(sig_cum_worst, 'Significance cumulative (ref: Worst Human)'))

    # Significance: LMM sensitivity (ordinal as continuous)
    sig_lmm_best = fit_ordinal_lmm(df_sig, reference_group='Best Human')
    sig_lmm_worst = fit_ordinal_lmm(df_sig, reference_group='Worst Human')
    all_results['significance_lmm_vs_best'] = sig_lmm_best
    all_results['significance_lmm_vs_worst'] = sig_lmm_worst
    report_lines.append(format_results(sig_lmm_best, 'Significance LMM sensitivity (ref: Best Human)'))
    report_lines.append(format_results(sig_lmm_worst, 'Significance LMM sensitivity (ref: Worst Human)'))

    # Evidence (binary)
    evi_best = fit_logistic_glmm(df_evi, 'evidence', reference_group='Best Human')
    evi_worst = fit_logistic_glmm(df_evi, 'evidence', reference_group='Worst Human')
    all_results['evidence_vs_best'] = evi_best
    all_results['evidence_vs_worst'] = evi_worst
    report_lines.append(format_results(evi_best, 'Evidence (ref: Best Human)'))
    report_lines.append(format_results(evi_worst, 'Evidence (ref: Worst Human)'))

    # ICC summary
    report_lines.append("\n" + "=" * 80)
    report_lines.append("SUMMARY: ICC VALUES")
    report_lines.append("=" * 80)
    for key, res in all_results.items():
        if 'cumulative' in key:
            for label, sub in res.get('cutpoints', {}).items():
                icc = sub.get('icc', np.nan)
                if not (isinstance(icc, float) and np.isnan(icc)):
                    report_lines.append(f"  {key}.{label:<25} ICC = {icc:.4f}  ({icc * 100:.1f}%)")
        else:
            icc = res.get('icc', np.nan)
            if not (isinstance(icc, float) and np.isnan(icc)):
                note = f"  ({res['icc_note']})" if 'icc_note' in res else ""
                report_lines.append(f"  {key:<40} ICC = {icc:.4f}  ({icc * 100:.1f}%){note}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = overview_report + "\n" + "\n".join(report_lines)
    print(report)
    with open(output_path / 'glmm_report.txt', 'w') as f:
        f.write(report)
    with open(output_path / 'dataset_overview.json', 'w') as f:
        json.dump(overview, f, indent=2, default=str)

    for key, res in all_results.items():
        if 'cumulative' in key:
            for label, sub in res.get('cutpoints', {}).items():
                with open(output_path / f'glmm_{key}_{label}_summary.txt', 'w') as f:
                    f.write(sub.get('summary', 'no summary'))
        else:
            with open(output_path / f'glmm_{key}_summary.txt', 'w') as f:
                f.write(res.get('summary', 'no summary'))

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
