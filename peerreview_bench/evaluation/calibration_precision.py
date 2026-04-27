#!/usr/bin/env python3
"""
Calibrate PeerReview Bench precision scores using the 908-item meta-review
bench as a calibration set (where we have both GPT-5.4 meta-reviewer and
human expert judgments).

Four calibration methods:
  1. GLM debiasing (AlpacaEval-style): logistic regression with confounders
  2. Rogan-Gladen prevalence correction on the "fully good" rate
  3. Per-axis Platt scaling + cascade
  4. Direct paper-level regression

Usage:
    python3 calibration_precision.py
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_FORCE_REDOWNLOAD', '0')

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent
sys.path.insert(0, str(_BENCH))

from load_data import load_meta_reviewer

NORM = {
    'claude-opus-4-5': 'Claude', 'review_claude-opus-4-7': 'Claude',
    'gpt-5.2': 'GPT',
    'gemini-3.0-pro-preview': 'Gemini', 'review_gemini-3.1-pro-preview': 'Gemini',
    'Human_1': 'Human_1', 'Human_2': 'Human_2', 'Human_3': 'Human_3',
}

EVAL_DIR = _BENCH / 'outputs' / 'eval'


# ============================================================================
# Load calibration data (908-item meta-review bench)
# ============================================================================

def load_calibration_data():
    """Load paired (meta-reviewer prediction, expert ground truth) from the
    908-item meta-review bench."""
    rows = load_meta_reviewer()

    results_dir = _BENCH / 'metareview_bench' / 'results' / 'litellm_proxy__azure_ai__gpt_5_4'
    predictions = {}
    for fname in os.listdir(results_dir):
        fpath = results_dir / fname
        if fpath.is_dir() or not fname.endswith('_metareview.json'):
            continue
        with open(fpath) as f:
            data = json.load(f)
        pid = data['paper_id']
        for reviewer in data['reviewers']:
            rid = NORM.get(reviewer['reviewer_id'], reviewer['reviewer_id'])
            for item in reviewer['items']:
                predictions[(pid, rid, item['item_number'])] = item

    paired = []
    for r in rows:
        k = (r['paper_id'], r['reviewer_id'], r['review_item_number'])
        pred = predictions.get(k)
        if not pred:
            continue

        pc = pred.get('correctness')
        ps = pred.get('significance')
        pe = pred.get('evidence')
        gc = r.get('correctness_primary')
        gs = r.get('significance_primary')
        ge = r.get('evidence_primary')

        if not pc or not gc:
            continue

        meta_fg = (pc == 'Correct' and ps == 'Significant' and pe == 'Sufficient')
        expert_fg = (gc == 'Correct' and gs == 'Significant' and ge == 'Sufficient')

        paired.append({
            'paper_id': r['paper_id'],
            'reviewer_id': r['reviewer_id'],
            'reviewer_type': r['reviewer_type'],
            'item_number': r['review_item_number'],
            'meta_correct': int(pc == 'Correct'),
            'meta_sig': int(ps == 'Significant'),
            'meta_suff': int(pe == 'Sufficient') if pe else 0,
            'meta_fg': int(meta_fg),
            'expert_correct': int(gc == 'Correct'),
            'expert_sig': int(gs == 'Significant'),
            'expert_suff': int(ge == 'Sufficient') if ge else 0,
            'expert_fg': int(expert_fg),
            'is_same_family': int(r['reviewer_id'] == 'GPT'),  # GPT-5.4 judging GPT
            'is_ai': int(r['reviewer_type'] == 'AI'),
        })

    return paired


# ============================================================================
# Load PeerReview Bench precision results
# ============================================================================

def load_precision_results():
    """Load all precision result JSONs from outputs/eval/."""
    results = {}
    for f in EVAL_DIR.glob('precision_*.json'):
        model = f.stem.replace('precision_', '')
        with open(f) as fh:
            data = json.load(fh)
        results[model] = data
    return results


# ============================================================================
# Method 1: GLM debiasing (AlpacaEval-style)
# ============================================================================

def method1_glm(paired, precision_results):
    """Fit logistic regression: P(expert_fg) ~ meta_fg + is_same_family + is_ai.
    Then predict debiased P(expert_fg) for each item with confounders zeroed out."""
    from sklearn.linear_model import LogisticRegression

    X = np.array([[p['meta_fg'], p['is_same_family'], p['is_ai']] for p in paired])
    y = np.array([p['expert_fg'] for p in paired])

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    coefs = dict(zip(['meta_fg', 'is_same_family', 'is_ai'], model.coef_[0]))
    print(f"  GLM coefficients: {coefs}")
    print(f"  GLM intercept: {model.intercept_[0]:.3f}")

    # Calibration: for each meta_fg=1 item, predict P(expert_fg) with confounders=0
    # P(expert_fg | meta_fg=1, same_family=0, is_ai=0)
    p_fg_debiased = model.predict_proba([[1, 0, 0]])[0][1]
    p_nfg_debiased = model.predict_proba([[0, 0, 0]])[0][1]

    print(f"  P(expert_fg | meta=FG, no confounders) = {p_fg_debiased:.3f}")
    print(f"  P(expert_fg | meta=not-FG, no confounders) = {p_nfg_debiased:.3f}")

    # Apply to precision results
    calibrated = {}
    for model_name, data in precision_results.items():
        items = data['per_item']
        n_meta_fg = sum(1 for i in items if i['is_fully_good'])
        n_meta_nfg = len(items) - n_meta_fg
        # Expected number of truly FG items
        expected_fg = n_meta_fg * p_fg_debiased + n_meta_nfg * p_nfg_debiased
        calibrated_precision = expected_fg / len(items) if items else 0
        calibrated[model_name] = calibrated_precision

    return calibrated


# ============================================================================
# Method 2: Rogan-Gladen prevalence correction
# ============================================================================

def method2_rogan_gladen(paired, precision_results):
    """Apply Rogan-Gladen correction using sensitivity/specificity of the
    meta-reviewer's 'fully good' classification."""

    meta_fg = np.array([p['meta_fg'] for p in paired])
    expert_fg = np.array([p['expert_fg'] for p in paired])

    # Sensitivity = P(meta=FG | expert=FG)
    expert_fg_mask = expert_fg == 1
    sensitivity = meta_fg[expert_fg_mask].mean() if expert_fg_mask.sum() > 0 else 0

    # Specificity = P(meta≠FG | expert≠FG)
    expert_nfg_mask = expert_fg == 0
    specificity = (1 - meta_fg[expert_nfg_mask]).mean() if expert_nfg_mask.sum() > 0 else 0

    print(f"  Sensitivity (P(meta=FG|expert=FG)): {sensitivity:.3f}")
    print(f"  Specificity (P(meta≠FG|expert≠FG)): {specificity:.3f}")
    print(f"  Denominator (Se + Sp - 1): {sensitivity + specificity - 1:.3f}")

    calibrated = {}
    for model_name, data in precision_results.items():
        raw = data['precision']
        denom = sensitivity + specificity - 1
        if abs(denom) < 0.01:
            corrected = raw  # correction unstable
            print(f"  WARNING: {model_name} — denominator too small, skipping correction")
        else:
            corrected = (raw + specificity - 1) / denom
            corrected = max(0, min(1, corrected))
        calibrated[model_name] = corrected

    return calibrated, sensitivity, specificity


# ============================================================================
# Method 3: Per-axis Platt scaling + cascade
# ============================================================================

def method3_platt_scaling(paired, precision_results):
    """Fit per-axis logistic regression (Platt scaling) on the calibration set,
    then apply calibrated probabilities to estimate true FG rate."""
    from sklearn.linear_model import LogisticRegression

    # Axis 1: Correctness
    X_corr = np.array([[p['meta_correct']] for p in paired])
    y_corr = np.array([p['expert_correct'] for p in paired])
    lr_corr = LogisticRegression(max_iter=1000).fit(X_corr, y_corr)

    p_correct_given_meta_correct = lr_corr.predict_proba([[1]])[0][1]
    p_correct_given_meta_notcorrect = lr_corr.predict_proba([[0]])[0][1]

    # Axis 2: Significance (among items where meta says Correct)
    sig_items = [p for p in paired if p['meta_correct'] == 1 and p['expert_correct'] == 1]
    if sig_items:
        X_sig = np.array([[p['meta_sig']] for p in sig_items])
        y_sig = np.array([p['expert_sig'] for p in sig_items])
        if len(set(y_sig)) > 1:
            lr_sig = LogisticRegression(max_iter=1000).fit(X_sig, y_sig)
            p_sig_given_meta_sig = lr_sig.predict_proba([[1]])[0][1]
        else:
            p_sig_given_meta_sig = y_sig.mean()
    else:
        p_sig_given_meta_sig = 0.5

    # Axis 3: Evidence (among items where meta says Correct + Significant)
    evi_items = [p for p in paired if p['meta_correct'] == 1 and p['meta_sig'] == 1
                 and p['expert_correct'] == 1 and p['expert_sig'] == 1]
    if evi_items:
        X_evi = np.array([[p['meta_suff']] for p in evi_items])
        y_evi = np.array([p['expert_suff'] for p in evi_items])
        if len(set(y_evi)) > 1:
            lr_evi = LogisticRegression(max_iter=1000).fit(X_evi, y_evi)
            p_suff_given_meta_suff = lr_evi.predict_proba([[1]])[0][1]
        else:
            p_suff_given_meta_suff = y_evi.mean()
    else:
        p_suff_given_meta_suff = 0.5

    print(f"  P(expert_correct | meta_correct) = {p_correct_given_meta_correct:.3f}")
    print(f"  P(expert_sig | meta_sig, both_correct) = {p_sig_given_meta_sig:.3f}")
    print(f"  P(expert_suff | meta_suff, both_correct+sig) = {p_suff_given_meta_suff:.3f}")

    # Cascade: P(FG) = P(correct) × P(sig|correct) × P(suff|correct,sig)
    p_fg_calibrated = p_correct_given_meta_correct * p_sig_given_meta_sig * p_suff_given_meta_suff
    print(f"  Calibrated P(FG | meta=FG) = {p_fg_calibrated:.3f}")

    # Apply: for each model, scale the meta-reviewer's FG count
    calibrated = {}
    for model_name, data in precision_results.items():
        items = data['per_item']
        n_meta_fg = sum(1 for i in items if i['is_fully_good'])
        # Each meta-FG item has P(truly FG) = p_fg_calibrated
        expected_fg = n_meta_fg * p_fg_calibrated
        calibrated_precision = expected_fg / len(items) if items else 0
        calibrated[model_name] = calibrated_precision

    return calibrated


# ============================================================================
# Method 4: Direct paper-level regression
# ============================================================================

def method4_paper_regression(paired, precision_results):
    """Fit linear regression from meta-reviewer per-paper FG rate to expert
    per-paper FG rate on the 27 calibration papers. Apply to 78 eval papers."""
    from sklearn.linear_model import LinearRegression

    # Aggregate to paper level
    paper_meta = defaultdict(list)
    paper_expert = defaultdict(list)
    for p in paired:
        paper_meta[p['paper_id']].append(p['meta_fg'])
        paper_expert[p['paper_id']].append(p['expert_fg'])

    papers = sorted(paper_meta.keys())
    X = np.array([[np.mean(paper_meta[pid])] for pid in papers])
    y = np.array([np.mean(paper_expert[pid]) for pid in papers])

    lr = LinearRegression().fit(X, y)
    print(f"  Regression: expert_FG = {lr.coef_[0]:.3f} × meta_FG + {lr.intercept_:.3f}")
    print(f"  R² on calibration set: {lr.score(X, y):.3f}")
    print(f"  N calibration papers: {len(papers)}")

    # Apply to precision results
    calibrated = {}
    for model_name, data in precision_results.items():
        raw = data['precision']
        corrected = lr.predict([[raw]])[0]
        corrected = max(0, min(1, corrected))
        calibrated[model_name] = corrected

    return calibrated


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading calibration data (908-item meta-review bench)...")
    paired = load_calibration_data()
    print(f"  Loaded {len(paired)} paired items")

    print("\nLoading PeerReview Bench precision results...")
    precision_results = load_precision_results()
    for m, d in sorted(precision_results.items()):
        print(f"  {m}: raw precision = {d['precision']*100:.2f}% ({d['n_fully_good']}/{d['n_items']})")

    # Table 5 reference values (expert-judged FG rates, paper-level)
    table5_expert = {
        'gpt-5.2': 60.0,
        'claude-opus-4-5': 53.1,
        'gemini-3.0-pro-preview': 50.2,  # if available
    }

    print("\n" + "=" * 80)
    print("METHOD 1: GLM Debiasing (AlpacaEval-style)")
    print("=" * 80)
    cal1 = method1_glm(paired, precision_results)

    print("\n" + "=" * 80)
    print("METHOD 2: Rogan-Gladen Prevalence Correction")
    print("=" * 80)
    cal2, sens, spec = method2_rogan_gladen(paired, precision_results)

    print("\n" + "=" * 80)
    print("METHOD 3: Per-Axis Platt Scaling + Cascade")
    print("=" * 80)
    cal3 = method3_platt_scaling(paired, precision_results)

    print("\n" + "=" * 80)
    print("METHOD 4: Direct Paper-Level Regression")
    print("=" * 80)
    cal4 = method4_paper_regression(paired, precision_results)

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: Raw vs Calibrated Precision")
    print("=" * 80)
    print(f"\n{'Model':<25} {'Raw':>7} {'GLM':>7} {'R-G':>7} {'Platt':>7} {'Regr':>7} {'Expert':>7}")
    print("-" * 75)
    for model in sorted(precision_results.keys()):
        raw = precision_results[model]['precision'] * 100
        c1 = cal1.get(model, 0) * 100
        c2 = cal2.get(model, 0) * 100
        c3 = cal3.get(model, 0) * 100
        c4 = cal4.get(model, 0) * 100
        expert = table5_expert.get(model, None)
        expert_str = f"{expert:.1f}%" if expert else "—"
        print(f"{model:<25} {raw:>6.1f}% {c1:>6.1f}% {c2:>6.1f}% {c3:>6.1f}% {c4:>6.1f}% {expert_str:>7}")

    # For models with expert data, compute MAE
    print(f"\nCalibration error (MAE vs expert FG rate, models with expert data):")
    for method_name, cal in [("GLM", cal1), ("R-G", cal2), ("Platt", cal3), ("Regr", cal4)]:
        errors = []
        for model, expert_val in table5_expert.items():
            if model in cal:
                errors.append(abs(cal[model] * 100 - expert_val))
        if errors:
            print(f"  {method_name}: MAE = {np.mean(errors):.1f}pp (on {len(errors)} models)")


if __name__ == '__main__':
    main()
