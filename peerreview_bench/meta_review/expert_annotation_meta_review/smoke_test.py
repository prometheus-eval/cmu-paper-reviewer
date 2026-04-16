#!/usr/bin/env python3
"""
Smoke test for every meta-reviewer target × both prompt modes.

Runs 3 real rows from the `meta_reviewer` HF config through:
  1. Baselines (majority, random) — instant, no network
  2. Every LLM in the model list — via the LiteLLM proxy

For each model, both `axis` and `tenclass` modes are tested (6 predictions
per model). The test prints parsed predictions + per-axis accuracy on the
3 rows so you can sanity-check calibration immediately.

Agent baselines are NOT covered here — they're slow (minutes per item)
and require a local PAPER_ROOT. Use `run_meta_review_agent.sh --limit 1`
for agent smoke testing.

Usage:
    cd meta_review/expert_annotation_meta_review
    python3 smoke_test.py                    # all models, 3 rows
    python3 smoke_test.py --limit 5          # 5 rows per model
    python3 smoke_test.py --models gemini    # substring match on model name
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_META_REVIEW_DIR = _HERE.parent
_BENCH_DIR = _META_REVIEW_DIR.parent

for p in (_HERE, _META_REVIEW_DIR, _BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

from load_data import load_meta_reviewer  # noqa: E402
from predictors import get_predictor, BASELINE_REGISTRY  # noqa: E402
from prompts import axis_to_tenclass_label, TENCLASS_LABEL_TO_ID  # noqa: E402
from metrics import (  # noqa: E402
    evaluate_axis_predictions,
    evaluate_tenclass_predictions,
    format_mode_report,
)


# Keep in sync with run_meta_review.sh
BASELINE_MODELS = ["majority", "random"]

LLM_MODELS = [
    "litellm_proxy/azure_ai/gpt-5.4",
    "litellm_proxy/azure_ai/grok-4-1-fast-reasoning",
    "litellm_proxy/azure_ai/Kimi-K2.5",
    "litellm_proxy/gemini/gemini-3.1-pro-preview",
    "litellm_proxy/anthropic/claude-opus-4-6",
    "litellm_proxy/fireworks_ai/accounts/fireworks/models/qwen3p6-plus",
]

MODES = ["axis", "tenclass"]


def _slug(name: str) -> str:
    return name.replace("/", "_").replace(".", "_").replace(" ", "_")


def _print_axis_pred(idx, pred):
    parsed = pred.get('_parsed', True)
    tag = 'OK' if parsed else 'FAIL'
    corr = pred.get('correctness', '?')
    sig = pred.get('significance', '?')
    evi = pred.get('evidence', '?')
    secs = pred.get('_elapsed_seconds', 0)
    imgs = pred.get('_n_images', 0)
    arts = pred.get('_n_artifacts', 0)
    reason_preview = (pred.get('reasoning') or '')[:80].replace('\n', ' ')
    print(f'  [{idx}] {tag:<5} corr={corr:<14} sig={str(sig):<25} evi={str(evi):<15} '
          f'({secs:.1f}s, {imgs}img, {arts}art)')
    print(f'         reasoning={reason_preview!r}')


def _print_tenclass_pred(idx, pred):
    parsed = pred.get('_parsed', True)
    tag = 'OK' if parsed else 'FAIL'
    label = pred.get('label', '?')
    label_id = pred.get('label_id', '?')
    secs = pred.get('_elapsed_seconds', 0)
    reason_preview = (pred.get('reasoning') or '')[:80].replace('\n', ' ')
    print(f'  [{idx}] {tag:<5} label={label_id}:{label}  ({secs:.1f}s)')
    print(f'         reasoning={reason_preview!r}')


def run_smoke(model: str, rows: list, mode: str) -> dict:
    """Run one model × one mode on the given rows. Returns a summary dict."""
    print(f'\n  --- {mode} mode ---')
    try:
        predictor = get_predictor(
            model,
            mode=mode,
            attach_images=False,   # skip images for speed in smoke test
            attach_artifacts=False,
        )
    except Exception as e:
        print(f'  FAILED to instantiate: {type(e).__name__}: {e}')
        return {'status': 'instantiation_failed', 'error': str(e), 'mode': mode}

    preds = []
    n_ok = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        try:
            pred = predictor.predict(row)
        except Exception as e:
            print(f'  [{i}] EXCEPTION: {type(e).__name__}: {e}')
            pred = {'_parsed': False, '_error': str(e), 'reasoning': f'(exception: {e})'}
            if mode == 'axis':
                pred.update({'correctness': 'Correct', 'significance': 'Significant', 'evidence': 'Sufficient'})
            else:
                pred.update({'label': 'correct_significant_sufficient', 'label_id': 1})
        preds.append(pred)
        if mode == 'axis':
            _print_axis_pred(i, pred)
        else:
            _print_tenclass_pred(i, pred)
        if pred.get('_parsed', True):
            n_ok += 1
    total_time = time.time() - t0

    # Quick scoring
    if mode == 'axis':
        metrics = evaluate_axis_predictions(preds, rows)
    else:
        metrics = evaluate_tenclass_predictions(preds, rows)
    report = format_mode_report(metrics, predictor_name=model)
    print(report)

    return {
        'status': 'ok' if n_ok == len(rows) else 'partial' if n_ok else 'failed',
        'mode': mode,
        'n_ok': n_ok,
        'n_total': len(rows),
        'elapsed_seconds': round(total_time, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Meta-review smoke test')
    parser.add_argument('--limit', type=int, default=3,
                        help='Rows per model (default 3)')
    parser.add_argument('--models', type=str, default=None,
                        help='Substring filter on model name (e.g. "gemini")')
    parser.add_argument('--mode', type=str, default=None, choices=MODES,
                        help='Only test one mode (default: both)')
    args = parser.parse_args()

    print('Loading meta_reviewer config...')
    rows = load_meta_reviewer()
    rows = rows[:args.limit]
    print(f'  {len(rows)} rows for smoke test')

    modes = [args.mode] if args.mode else MODES
    models = BASELINE_MODELS + LLM_MODELS
    if args.models:
        models = [m for m in models if args.models.lower() in m.lower()]
    print(f'  models: {len(models)}')
    print(f'  modes: {modes}')

    summary = {}
    for model in models:
        print('\n' + '=' * 80)
        print(f'  {model}')
        print('=' * 80)
        model_results = {}
        for mode in modes:
            result = run_smoke(model, rows, mode)
            model_results[mode] = result
        summary[model] = model_results

    # Final summary table
    print('\n' + '=' * 80)
    print('SMOKE TEST SUMMARY')
    print('=' * 80)
    print(f"{'model':<62} {'mode':<10} {'status':<10} {'ok':>4} {'time':>8}")
    print('-' * 100)
    for model, results in summary.items():
        for mode, r in results.items():
            status = r.get('status', '?')
            n_ok = r.get('n_ok', 0)
            n_total = r.get('n_total', 0)
            elapsed = r.get('elapsed_seconds', 0)
            print(f'{model:<62} {mode:<10} {status:<10} {n_ok}/{n_total}  {elapsed:>6.1f}s')

    # Write summary
    out_dir = _BENCH_DIR / 'outputs' / 'expert_annotation_meta_review' / 'smoke_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nSummary written to {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
