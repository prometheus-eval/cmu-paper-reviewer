"""
Evaluation script for similarity_check baselines.

Takes a baseline output JSON (from embedding_classifier.py or
llm_classifier.py) and reports:
  - Binary accuracy / precision / recall / F1
  - AUROC and AUPRC (for embedding scores)
  - Confusion matrix per finegrained sub-label (one of the four
    long-form labels used in the HF schema)
  - Per-pair-type breakdown (AI-AI vs AI-Human)

Usage:
    python evaluate.py outputs/embedding_qwen3.json
    python evaluate.py outputs/llm_judge_4way_openai_gpt-5.2-2025-12-11.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from prompts import fourway_to_binary  # noqa: E402


FINEGRAINED_LABELS = (
    'same subject, same argument, same evidence',
    'same subject, same argument, different evidence',
    'same subject, different argument',
    'different subject',
)


def has_score(record: Dict) -> bool:
    return 'cosine_score' in record


def has_parsed_answer(record: Dict) -> bool:
    return 'parsed_answer' in record


def evaluate_embedding(records: List[Dict], threshold: Optional[float] = None) -> Dict:
    """Compute ROC + best-threshold metrics for embedding scores."""
    import numpy as np
    scores = np.array([r['cosine_score'] for r in records])
    labels = np.array([r['binary_label'] for r in records])
    y_true = (labels == 'similar').astype(int)

    # AUROC and AUPRC via simple sklearn
    try:
        from sklearn.metrics import (roc_auc_score, average_precision_score,
                                      precision_recall_curve, roc_curve)
        auroc = float(roc_auc_score(y_true, scores))
        auprc = float(average_precision_score(y_true, scores))
        # Find threshold that maximizes F1
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, scores)
        f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12)
        best_idx = f1_arr[:-1].argmax()  # last entry is the no-positive sentinel
        best_thr = float(thr_arr[best_idx])
        best_f1 = float(f1_arr[best_idx])
        best_prec = float(prec_arr[best_idx])
        best_rec = float(rec_arr[best_idx])
    except ImportError:
        print('  (sklearn not available — AUROC / AUPRC skipped)')
        auroc = auprc = None
        best_thr = best_f1 = best_prec = best_rec = None

    # If a threshold was provided, also report metrics there
    if threshold is None and best_thr is not None:
        threshold = best_thr

    if threshold is not None:
        y_pred = (scores >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        acc = (tp + tn) / max(1, tp + fp + fn + tn)
    else:
        tp = fp = fn = tn = 0
        acc = None

    # Per-finegrained mean cosine score
    by_cat = {}
    for cat in FINEGRAINED_LABELS:
        cat_records = [r for r in records if r['finegrained_label'] == cat]
        if cat_records:
            vals = np.array([r['cosine_score'] for r in cat_records])
            by_cat[cat] = {
                'n': len(cat_records),
                'mean_score': float(vals.mean()),
                'std_score': float(vals.std()),
            }

    # Per-pair-type mean
    by_type = {}
    for pt in ('AI-AI', 'AI-Human'):
        pt_records = [r for r in records if r['pair_type'] == pt]
        if pt_records:
            vals = np.array([r['cosine_score'] for r in pt_records])
            sim = np.array([1 if r['binary_label'] == 'similar' else 0 for r in pt_records])
            try:
                from sklearn.metrics import roc_auc_score
                pt_auroc = float(roc_auc_score(sim, vals))
            except Exception:
                pt_auroc = None
            by_type[pt] = {
                'n': len(pt_records),
                'mean_score': float(vals.mean()),
                'auroc': pt_auroc,
            }

    return {
        'metric_type': 'embedding',
        'auroc': auroc,
        'auprc': auprc,
        'best_threshold': best_thr,
        'best_f1': best_f1,
        'best_precision': best_prec,
        'best_recall': best_rec,
        'best_accuracy': acc,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'per_finegrained_mean_score': by_cat,
        'per_pair_type': by_type,
    }


def evaluate_llm(records: List[Dict], mode: str) -> Dict:
    """Compute accuracy + confusion matrix for LLM-judge outputs."""
    n_total = len(records)
    n_parsed = sum(1 for r in records if r.get('parsed_answer') is not None)

    # Accuracy under each labeling scheme
    binary_labels = []
    binary_preds = []
    fourway_labels = []
    fourway_preds = []

    for r in records:
        parsed = r.get('parsed_answer')
        if parsed is None:
            continue
        binary_labels.append(r['binary_label'])
        fourway_labels.append(r['finegrained_label'])
        if mode == 'binary':
            binary_preds.append(parsed)
            fourway_preds.append(None)
        else:  # 4way
            fourway_preds.append(parsed)
            binary_preds.append(fourway_to_binary(parsed))

    n = len(binary_labels)
    binary_acc = sum(p == l for p, l in zip(binary_preds, binary_labels)) / max(1, n)

    # Per-class accuracy (both labels)
    per_binary = {}
    for label in ('similar', 'not_similar'):
        idxs = [i for i, l in enumerate(binary_labels) if l == label]
        if idxs:
            correct = sum(1 for i in idxs if binary_preds[i] == label)
            per_binary[label] = {
                'n': len(idxs),
                'accuracy': correct / len(idxs),
            }

    # Confusion matrix (binary)
    cm = Counter()
    for p, l in zip(binary_preds, binary_labels):
        cm[(l, p)] += 1
    confusion = {
        'true_similar_pred_similar': cm[('similar', 'similar')],
        'true_similar_pred_not_similar': cm[('similar', 'not_similar')],
        'true_not_similar_pred_similar': cm[('not_similar', 'similar')],
        'true_not_similar_pred_not_similar': cm[('not_similar', 'not_similar')],
    }

    # Per-finegrained accuracy
    per_finegrained = {}
    for cat in FINEGRAINED_LABELS:
        idxs = [i for i, l in enumerate(fourway_labels) if l == cat]
        if not idxs:
            continue
        per_finegrained[cat] = {'n': len(idxs)}
        if mode == '4way':
            correct = sum(1 for i in idxs if fourway_preds[i] == cat)
            per_finegrained[cat]['fourway_accuracy'] = correct / len(idxs)
        # Also report binary accuracy on this sub-bucket
        binary_truth = fourway_to_binary(cat)
        binary_correct = sum(1 for i in idxs if binary_preds[i] == binary_truth)
        per_finegrained[cat]['binary_accuracy'] = binary_correct / len(idxs)

    # Per pair-type accuracy
    per_pt = {}
    for pt in ('AI-AI', 'AI-Human'):
        idxs = [i for i, r in enumerate(records)
                if r['pair_type'] == pt and r.get('parsed_answer') is not None]
        if idxs:
            # We need to remap idxs to positions in the binary_* lists
            # Simpler: recompute directly
            ll = [records[i]['binary_label'] for i in idxs]
            if mode == 'binary':
                pp = [records[i]['parsed_answer'] for i in idxs]
            else:
                pp = [fourway_to_binary(records[i]['parsed_answer']) for i in idxs]
            acc = sum(p == l for p, l in zip(pp, ll)) / len(ll)
            per_pt[pt] = {'n': len(idxs), 'binary_accuracy': acc}

    return {
        'metric_type': 'llm_judge',
        'mode': mode,
        'n_total': n_total,
        'n_parsed': n_parsed,
        'parse_rate': n_parsed / max(1, n_total),
        'binary_accuracy': binary_acc,
        'per_binary_label': per_binary,
        'per_finegrained_label': per_finegrained,
        'per_pair_type': per_pt,
        'confusion_matrix': confusion,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a similarity baseline')
    parser.add_argument('input_path', type=Path,
                        help='Path to a baseline output JSON')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold for embedding metrics (default: best F1)')
    args = parser.parse_args()

    data = json.loads(args.input_path.read_text())
    records = data['results']
    metadata = data.get('metadata', {})

    print(f'Input: {args.input_path}')
    print(f'Metadata: {json.dumps(metadata, indent=2)}')
    print()

    # Detect type
    if records and has_score(records[0]):
        report = evaluate_embedding(records, args.threshold)
    elif records and has_parsed_answer(records[0]):
        mode = metadata.get('mode', 'binary')
        report = evaluate_llm(records, mode)
    else:
        raise ValueError('Could not detect baseline type from records')

    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
