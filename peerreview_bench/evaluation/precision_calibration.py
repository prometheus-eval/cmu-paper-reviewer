#!/usr/bin/env python3
"""
Calibrate PeerReview Bench precision scores.

Uses a Ridge regression trained on the 27-paper meta-review bench calibration
set (81 paper×model data points) with three generalizable features:
  1. meta_corr_rate  — meta-reviewer's correctness rate per paper
  2. log_review_words — log of avg review item length (proxy; fixed at log(101))
  3. family_sig_fpr  — per-family significance false positive rate from calibration

The calibration is trained on the 27 overlap papers where both meta-reviewer
and human expert judgments exist. It is tested on the remaining 55 papers
in PeerReview Bench (test MAE = 1.9pp on 3 models with expert ground truth).

Usage:
    python3 precision_calibration.py ../outputs/eval/precision_gpt-5.2.json
    python3 precision_calibration.py ../outputs/eval/precision_*.json
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent
PARAMS_FILE = _HERE / 'calibration_params.json'
PAPERS_DIR = _BENCH / 'papers'


def load_calibration_params():
    with open(PARAMS_FILE) as f:
        return json.load(f)


def get_family(model_name: str) -> str:
    m = model_name.lower()
    if 'gpt' in m:
        return 'GPT'
    elif 'claude' in m:
        return 'Claude'
    else:
        return 'Gemini'


def calibrate_precision(precision_json_path: str, params: dict = None) -> dict:
    """Calibrate a precision result file.

    Returns dict with raw_precision, calibrated_precision, per_paper details.
    """
    if params is None:
        params = load_calibration_params()

    with open(precision_json_path) as f:
        data = json.load(f)

    model_name = data.get('ai_reviewer_model', '')
    family = get_family(model_name)
    family_fpr = params['family_sig_fpr'].get(family, 0.4)

    scaler_mean = np.array(params['scaler_mean'])
    scaler_scale = np.array(params['scaler_scale'])
    coef = np.array(params['ridge_coef'])
    intercept = params['ridge_intercept']

    items = data['per_item']
    raw_precision = data['precision']

    # Group items by paper
    by_paper = defaultdict(list)
    by_paper_corr = defaultdict(list)
    for item in items:
        pid = item['paper_id']
        by_paper[pid].append(item['is_fully_good'])
        by_paper_corr[pid].append(1 if item.get('correctness') == 'Correct' else 0)

    # Calibrate per paper
    paper_results = []
    for pid in sorted(by_paper.keys()):
        fgs = by_paper[pid]
        meta_corr_rate = np.mean(by_paper_corr[pid])
        log_review_words = np.log1p(100)  # approximate default

        feat = np.array([meta_corr_rate, log_review_words, family_fpr])
        feat_scaled = (feat - scaler_mean) / scaler_scale
        cal = float(np.clip(feat_scaled @ coef + intercept, 0, 1))

        paper_results.append({
            'paper_id': pid,
            'n_items': len(fgs),
            'raw_fg_rate': np.mean(fgs),
            'calibrated_fg_rate': cal,
            'meta_corr_rate': meta_corr_rate,
        })

    calibrated_precision = np.mean([p['calibrated_fg_rate'] for p in paper_results])

    return {
        'model': model_name,
        'family': family,
        'raw_precision': raw_precision,
        'calibrated_precision': calibrated_precision,
        'n_papers': len(paper_results),
        'n_items': len(items),
        'per_paper': paper_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate PeerReview Bench precision scores')
    parser.add_argument('precision_files', nargs='+', type=str,
                        help='Path(s) to precision result JSON file(s)')
    args = parser.parse_args()

    params = load_calibration_params()

    print(f"{'Model':<25} {'Raw':>10} {'Calibrated':>12} {'Papers':>8} {'Items':>7}")
    print('-' * 65)

    for fpath in sorted(args.precision_files):
        result = calibrate_precision(fpath, params)
        print(f"{result['model']:<25} "
              f"{result['raw_precision']*100:>9.2f}% "
              f"{result['calibrated_precision']*100:>11.2f}% "
              f"{result['n_papers']:>8} "
              f"{result['n_items']:>7}")


if __name__ == '__main__':
    main()
