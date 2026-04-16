"""
Scoring a meta-reviewer's predictions against the `meta_reviewer` config.

The benchmark label is one of 10 classes that jointly encode:
  - the correctness / significance / evidence judgement (cascade aware)
  - whether expert meta-reviewers would agree on each axis

Given a list of (prediction, ground_truth_row) pairs, we compute:
  - top-line accuracy on the 10-class label
  - per-class precision / recall / F1
  - per-axis accuracy (correctness/significance/evidence, independent of agreement)
  - per-axis "agreement prediction" accuracy (computed against whether
    primary and secondary actually agreed)
  - confusion matrix over the 10 classes
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)


# ----------------------------------------------------------------------
# Module version / compatibility note
# ----------------------------------------------------------------------
#
# This module originally scored the legacy "with per-axis agreement
# fields" prediction format via `evaluate_predictions`. The new
# expert_annotation_meta_review code uses two cleaner prediction shapes:
#
#   AXIS mode     : {reasoning, correctness, significance, evidence}
#   TENCLASS mode : {reasoning, label, label_id}
#
# Use the new helpers `evaluate_axis_predictions` and
# `evaluate_tenclass_predictions` (added at the bottom of this file) for
# those formats. The old `evaluate_predictions` is kept for backward
# compatibility with any caller that still emits the 7-key shape.

META_LABEL_ID_TO_NAME = {
    1:  'correct_significant_sufficient',
    2:  'correct_significant_insufficient',
    3:  'correct_significant_disagree_on_evidence',
    4:  'correct_marginal_sufficient',
    5:  'correct_marginal_insufficient',
    6:  'correct_marginal_disagree_on_evidence',
    7:  'correct_not_significant',
    8:  'correct_disagree_on_significance',
    9:  'incorrect',
    10: 'disagree_on_correctness',
}
META_LABEL_NAME_TO_ID = {v: k for k, v in META_LABEL_ID_TO_NAME.items()}


# ----------------------------------------------------------------------
# Collapse a per-axis prediction into a 10-class label
# ----------------------------------------------------------------------

_SIG_TO_NUM = {
    'Not Significant': 0,
    'Marginally Significant': 1,
    'Significant': 2,
}
_EVI_TO_NUM = {
    'Requires More': 0,
    'Sufficient': 1,
}


def prediction_to_label_id(pred: Dict[str, Any]) -> int:
    """Collapse {correctness, significance, evidence, *_experts_agree} -> 1..10.

    Rules:
        - correctness_experts_agree == 'disagree'  ->  10 (disagree_on_correctness)
        - else if correctness == 'Not Correct'     ->  9  (incorrect)
        - else if significance_experts_agree == 'disagree' -> 8 (correct_disagree_on_significance)
        - else if significance == 'Not Significant'        -> 7 (correct_not_significant)
        - else (significance = Marg or Sig):
              sig_is_significant = (significance == 'Significant')
              if evidence_experts_agree == 'disagree'  -> 3 or 6
              else if evidence == 'Sufficient'         -> 1 or 4
              else (Requires More)                      -> 2 or 5
    """
    corr = pred.get('correctness')
    corr_agree = pred.get('correctness_experts_agree')
    sig = pred.get('significance')
    sig_agree = pred.get('significance_experts_agree')
    evi = pred.get('evidence')
    evi_agree = pred.get('evidence_experts_agree')

    if corr_agree == 'disagree':
        return 10
    if corr == 'Not Correct':
        return 9
    # corr == 'Correct' and experts agree
    if sig_agree == 'disagree':
        return 8
    if sig == 'Not Significant':
        return 7
    sig_is_significant = (sig == 'Significant')
    if evi_agree == 'disagree':
        return 3 if sig_is_significant else 6
    if evi == 'Sufficient':
        return 1 if sig_is_significant else 4
    return 2 if sig_is_significant else 5


# ----------------------------------------------------------------------
# Per-axis accuracy helpers (independent of agreement)
# ----------------------------------------------------------------------

def _consensus_label(row: Dict[str, Any], axis: str) -> Optional[str]:
    """Return the ground-truth string label for an axis on a row.

    We use the primary annotator's label as the reference ground truth when
    the two annotators agreed; if they disagreed we return None (axis is
    considered unanswerable for accuracy purposes)."""
    p_key = f'{axis}_primary'
    s_key = f'{axis}_secondary'
    p = row.get(p_key)
    s = row.get(s_key)
    if p is None or s is None:
        return None
    if p != s:
        return None
    return p


# ----------------------------------------------------------------------
# Main evaluation entry
# ----------------------------------------------------------------------

def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute metrics for the meta-review task.

    Args:
        predictions: list of dicts with keys {correctness, correctness_experts_agree,
            significance, significance_experts_agree, evidence, evidence_experts_agree}.
        ground_truth_rows: list of rows from the meta_reviewer HF config.

    Returns:
        A dict with top-line accuracy, per-class report, per-axis accuracy, etc.
    """
    assert len(predictions) == len(ground_truth_rows), 'length mismatch'
    n = len(predictions)
    if n == 0:
        return {'n': 0}

    # ---- 1. 10-class accuracy ----
    pred_ids: List[int] = []
    gt_ids: List[int] = []
    for pred, row in zip(predictions, ground_truth_rows):
        pred_ids.append(prediction_to_label_id(pred))
        gt_ids.append(int(row['label_id']))

    overall_acc = accuracy_score(gt_ids, pred_ids)

    labels_present = sorted(set(gt_ids) | set(pred_ids))
    macro_f1 = f1_score(gt_ids, pred_ids, labels=labels_present, average='macro', zero_division=0)
    weighted_f1 = f1_score(gt_ids, pred_ids, labels=labels_present, average='weighted', zero_division=0)

    per_class = classification_report(
        gt_ids, pred_ids,
        labels=labels_present,
        target_names=[META_LABEL_ID_TO_NAME[i] for i in labels_present],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(gt_ids, pred_ids, labels=labels_present).tolist()

    # ---- 2. Per-axis label accuracy (ignore rows where GT is ambiguous) ----
    axis_scores: Dict[str, Dict[str, Any]] = {}
    for axis in ('correctness', 'significance', 'evidence'):
        scored = 0
        correct = 0
        for pred, row in zip(predictions, ground_truth_rows):
            gt_value = _consensus_label(row, axis)
            if gt_value is None:
                continue  # skip rows with no consensus on this axis
            pred_value = pred.get(axis)
            if pred_value is None:
                # Predictor said N/A but GT has a value -> counted as incorrect
                scored += 1
                continue
            scored += 1
            if str(pred_value) == str(gt_value):
                correct += 1
        axis_scores[axis] = {
            'n_scored': scored,
            'n_correct': correct,
            'accuracy': (correct / scored) if scored > 0 else None,
        }

    # ---- 3. Per-axis agreement-prediction accuracy ----
    agreement_scores: Dict[str, Dict[str, Any]] = {}
    for axis in ('correctness', 'significance', 'evidence'):
        scored = 0
        correct = 0
        p_key, s_key = f'{axis}_primary', f'{axis}_secondary'
        for pred, row in zip(predictions, ground_truth_rows):
            p, s = row.get(p_key), row.get(s_key)
            if p is None or s is None:
                continue  # agreement not defined here
            scored += 1
            gt_agree = (p == s)
            pred_agree_raw = pred.get(f'{axis}_experts_agree')
            pred_agree = {'agree': True, 'disagree': False}.get(
                pred_agree_raw, None
            )
            if pred_agree is None:
                continue  # prediction missing
            if pred_agree == gt_agree:
                correct += 1
        agreement_scores[axis] = {
            'n_scored': scored,
            'n_correct': correct,
            'accuracy': (correct / scored) if scored > 0 else None,
        }

    return {
        'n': n,
        'overall_accuracy': overall_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'labels_present': [META_LABEL_ID_TO_NAME[i] for i in labels_present],
        'per_class_report': per_class,
        'confusion_matrix': cm,
        'label_accuracy_per_axis': axis_scores,
        'agreement_accuracy_per_axis': agreement_scores,
    }


def format_evaluation_report(metrics: Dict[str, Any], predictor_name: str = 'unknown') -> str:
    if metrics.get('n', 0) == 0:
        return f'No predictions to score (predictor={predictor_name}).'

    lines = []
    lines.append('=' * 80)
    lines.append(f'META-REVIEWER EVALUATION  (predictor: {predictor_name})')
    lines.append('=' * 80)
    lines.append(f"N rows scored:     {metrics['n']}")
    lines.append(f"10-class accuracy: {metrics['overall_accuracy']:.3f}")
    lines.append(f"Macro F1:          {metrics['macro_f1']:.3f}")
    lines.append(f"Weighted F1:       {metrics['weighted_f1']:.3f}")

    lines.append('')
    lines.append('Per-class precision / recall / F1 (on classes actually observed):')
    per_class = metrics.get('per_class_report', {})
    lines.append(f"  {'class':<45} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for name, vals in per_class.items():
        if name in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        if not isinstance(vals, dict):
            continue
        lines.append(
            f"  {name[:44]:<45} {vals['precision']:>10.3f} "
            f"{vals['recall']:>10.3f} {vals['f1-score']:>10.3f} {int(vals['support']):>10}"
        )

    lines.append('')
    lines.append('Per-axis LABEL accuracy (scored only on rows where both annotators agreed on that axis):')
    for axis, s in metrics.get('label_accuracy_per_axis', {}).items():
        if s['accuracy'] is None:
            lines.append(f"  {axis:<14} N/A (no scorable rows)")
        else:
            lines.append(
                f"  {axis:<14} {s['accuracy']:.3f}  ({s['n_correct']}/{s['n_scored']})"
            )

    lines.append('')
    lines.append('Per-axis AGREEMENT-prediction accuracy (predicting whether experts would agree):')
    for axis, s in metrics.get('agreement_accuracy_per_axis', {}).items():
        if s['accuracy'] is None:
            lines.append(f"  {axis:<14} N/A")
        else:
            lines.append(
                f"  {axis:<14} {s['accuracy']:.3f}  ({s['n_correct']}/{s['n_scored']})"
            )

    lines.append('')
    return '\n'.join(lines)


# ======================================================================
# New-format scoring: AXIS mode
# ======================================================================
#
# The axis-mode predictor emits {reasoning, correctness, significance,
# evidence}. We score it per-axis against the consensus ground-truth
# label (where primary and secondary annotators agreed). Rows where the
# two annotators disagreed on an axis are EXCLUDED from the accuracy
# denominator for that axis (they're ambiguous).
#
# Axis mode does NOT predict disagreement, so it cannot hit any of the
# "disagree" classes in the 10-class taxonomy (classes 3, 6, 8, 10).
# Any collapsed-label scoring we do against the 10-class ground truth
# will therefore systematically miss those four classes — we report a
# "reduced 6-class" accuracy that scores only on rows whose ground
# truth falls in {1, 2, 4, 5, 7, 9}.

_AXIS_REACHABLE_LABEL_IDS = (1, 2, 4, 5, 7, 9)


def evaluate_axis_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score axis-mode predictions.

    Args:
        predictions: list of dicts with keys {reasoning, correctness,
            significance, evidence}. `significance` and `evidence` may
            be None (cascade).
        ground_truth_rows: list of rows from either the meta_reviewer
            HF config (which has *_primary / *_secondary fields and a
            `label_id` column) or the expert_annotation HF config
            (which has `correctness` / `significance` / `evidence`
            single-annotator fields).

    Returns a dict with:
      - n: number of (pred, row) pairs
      - label_accuracy_per_axis: per-axis accuracy on consensus rows
      - reduced_6class_accuracy: accuracy against the 10-class GT,
        restricted to rows whose GT label falls in the 6 classes that
        axis mode is capable of predicting. None if the GT rows don't
        carry a label_id column.
      - per_class_report, confusion_matrix: on the reduced 6 classes
    """
    assert len(predictions) == len(ground_truth_rows), 'length mismatch'
    n = len(predictions)
    if n == 0:
        return {'n': 0, 'mode': 'axis'}

    # Per-axis accuracy on consensus labels
    axis_scores: Dict[str, Dict[str, Any]] = {}
    for axis in ('correctness', 'significance', 'evidence'):
        scored = 0
        correct = 0
        for pred, row in zip(predictions, ground_truth_rows):
            gt_value = _resolve_axis_gt(row, axis)
            if gt_value is None:
                continue
            scored += 1
            pred_value = pred.get(axis)
            if pred_value is None:
                continue  # cascade-null counts as incorrect
            if str(pred_value) == str(gt_value):
                correct += 1
        axis_scores[axis] = {
            'n_scored': scored,
            'n_correct': correct,
            'accuracy': (correct / scored) if scored > 0 else None,
        }

    # Reduced 6-class accuracy (only on rows whose GT label_id is in the
    # reachable set for axis-mode predictions).
    from prompts import axis_to_tenclass_label, TENCLASS_LABEL_TO_ID  # noqa: E402
    pred_ids: List[int] = []
    gt_ids: List[int] = []
    for pred, row in zip(predictions, ground_truth_rows):
        gt_id = _resolve_tenclass_label_id(row)
        if gt_id is None or gt_id not in _AXIS_REACHABLE_LABEL_IDS:
            continue
        pred_label = axis_to_tenclass_label(pred)
        pred_ids.append(TENCLASS_LABEL_TO_ID[pred_label])
        gt_ids.append(gt_id)

    reduced_result: Dict[str, Any] = {'n_scored': len(pred_ids)}
    if pred_ids:
        reduced_result['accuracy'] = float(accuracy_score(gt_ids, pred_ids))
        labels_present = sorted(set(gt_ids) | set(pred_ids))
        reduced_result['macro_f1'] = float(
            f1_score(gt_ids, pred_ids, labels=labels_present,
                     average='macro', zero_division=0)
        )
        reduced_result['weighted_f1'] = float(
            f1_score(gt_ids, pred_ids, labels=labels_present,
                     average='weighted', zero_division=0)
        )
        reduced_result['labels_present'] = [
            META_LABEL_ID_TO_NAME[i] for i in labels_present
        ]
        reduced_result['per_class_report'] = classification_report(
            gt_ids, pred_ids,
            labels=labels_present,
            target_names=[META_LABEL_ID_TO_NAME[i] for i in labels_present],
            output_dict=True,
            zero_division=0,
        )
        reduced_result['confusion_matrix'] = confusion_matrix(
            gt_ids, pred_ids, labels=labels_present
        ).tolist()
    else:
        reduced_result['accuracy'] = None

    return {
        'mode': 'axis',
        'n': n,
        'label_accuracy_per_axis': axis_scores,
        'reduced_6class': reduced_result,
    }


def _resolve_axis_gt(row: Dict[str, Any], axis: str) -> Optional[str]:
    """Ground-truth string label for an axis on a row.

    Supports two row schemas:
      - meta_reviewer config (or merged expert_annotation with both
        annotators): has `<axis>_primary` and `<axis>_secondary`.
        If both annotators are present and agree, return the consensus
        label. If both are present and disagree, return None (ambiguous).
        If only one annotator is present (secondary is None), use the
        primary label as single-annotator ground truth.
      - expert_annotation config (single annotator): has a bare `<axis>`
        field. Return it directly.

    Returns None if neither schema applies, or if annotators disagree.
    """
    p_key = f'{axis}_primary'
    s_key = f'{axis}_secondary'
    if p_key in row:
        p = row.get(p_key)
        s = row.get(s_key)
        if p is not None and s is not None:
            # Both annotators present
            if p != s:
                return None  # disagree → ambiguous
            return p  # agree → consensus
        if p is not None:
            # Only primary annotator → single-annotator ground truth
            return p
        if s is not None:
            return s
        # Both None → no ground truth
        # Fall through to bare-field check
    # expert_annotation fallback: single-annotator column
    v = row.get(axis)
    return v if v is not None else None


def _resolve_tenclass_label_id(row: Dict[str, Any]) -> Optional[int]:
    """Return the 10-class label_id for a row, or None if not derivable.

    - meta_reviewer config: `label_id` column is present — use it directly.
    - expert_annotation config: derive from primary/secondary where both
      exist. For rows with only one annotator, return None (no ground
      truth for the full 10-class taxonomy is possible).
    """
    if 'label_id' in row and row['label_id'] is not None:
        try:
            return int(row['label_id'])
        except (TypeError, ValueError):
            return None
    # Try deriving from primary/secondary
    try:
        from prompts import axis_to_tenclass_label, TENCLASS_LABEL_TO_ID  # noqa
    except Exception:
        return None
    corr_p = row.get('correctness_primary')
    corr_s = row.get('correctness_secondary')
    if corr_p is None or corr_s is None:
        return None
    if corr_p != corr_s:
        return TENCLASS_LABEL_TO_ID['disagree_on_correctness']
    if corr_p == 'Not Correct':
        return TENCLASS_LABEL_TO_ID['incorrect']
    # Both correct — check significance
    sig_p = row.get('significance_primary')
    sig_s = row.get('significance_secondary')
    if sig_p is None or sig_s is None:
        return None
    if sig_p != sig_s:
        return TENCLASS_LABEL_TO_ID['correct_disagree_on_significance']
    if sig_p == 'Not Significant':
        return TENCLASS_LABEL_TO_ID['correct_not_significant']
    sig_is_significant = (sig_p == 'Significant')
    # Check evidence
    evi_p = row.get('evidence_primary')
    evi_s = row.get('evidence_secondary')
    if evi_p is None or evi_s is None:
        return None
    if evi_p != evi_s:
        return (
            TENCLASS_LABEL_TO_ID['correct_significant_disagree_on_evidence']
            if sig_is_significant
            else TENCLASS_LABEL_TO_ID['correct_marginal_disagree_on_evidence']
        )
    if evi_p == 'Sufficient':
        return (
            TENCLASS_LABEL_TO_ID['correct_significant_sufficient']
            if sig_is_significant
            else TENCLASS_LABEL_TO_ID['correct_marginal_sufficient']
        )
    return (
        TENCLASS_LABEL_TO_ID['correct_significant_insufficient']
        if sig_is_significant
        else TENCLASS_LABEL_TO_ID['correct_marginal_insufficient']
    )


# ======================================================================
# New-format scoring: TENCLASS mode
# ======================================================================

def evaluate_tenclass_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score tenclass-mode predictions.

    Args:
        predictions: list of dicts with keys {reasoning, label, label_id}
        ground_truth_rows: list of rows from either meta_reviewer or
            expert_annotation HF config.

    Returns:
        dict with 10-class accuracy, macro/weighted F1, per-class report,
        confusion matrix. If ground truth isn't derivable for a row (e.g.,
        single-annotator row from expert_annotation), that row is
        excluded from the denominator.
    """
    assert len(predictions) == len(ground_truth_rows), 'length mismatch'
    n = len(predictions)
    if n == 0:
        return {'n': 0, 'mode': 'tenclass'}

    pred_ids: List[int] = []
    gt_ids: List[int] = []
    n_unscorable = 0
    for pred, row in zip(predictions, ground_truth_rows):
        pred_id = pred.get('label_id')
        if pred_id is None:
            lab = pred.get('label')
            pred_id = META_LABEL_NAME_TO_ID.get(lab)
        if pred_id is None:
            # Parse failure — skip this row
            n_unscorable += 1
            continue
        gt_id = _resolve_tenclass_label_id(row)
        if gt_id is None:
            n_unscorable += 1
            continue
        pred_ids.append(int(pred_id))
        gt_ids.append(int(gt_id))

    if not pred_ids:
        return {
            'mode': 'tenclass',
            'n': n,
            'n_scored': 0,
            'n_unscorable': n_unscorable,
            'overall_accuracy': None,
        }

    overall_acc = float(accuracy_score(gt_ids, pred_ids))
    labels_present = sorted(set(gt_ids) | set(pred_ids))
    macro_f1 = float(
        f1_score(gt_ids, pred_ids, labels=labels_present,
                 average='macro', zero_division=0)
    )
    weighted_f1 = float(
        f1_score(gt_ids, pred_ids, labels=labels_present,
                 average='weighted', zero_division=0)
    )
    per_class = classification_report(
        gt_ids, pred_ids,
        labels=labels_present,
        target_names=[META_LABEL_ID_TO_NAME[i] for i in labels_present],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(gt_ids, pred_ids, labels=labels_present).tolist()

    return {
        'mode': 'tenclass',
        'n': n,
        'n_scored': len(pred_ids),
        'n_unscorable': n_unscorable,
        'overall_accuracy': overall_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'labels_present': [META_LABEL_ID_TO_NAME[i] for i in labels_present],
        'per_class_report': per_class,
        'confusion_matrix': cm,
    }


def format_mode_report(metrics: Dict[str, Any], predictor_name: str = 'unknown') -> str:
    """Human-readable report for the new-format metrics dicts (both axis
    and tenclass)."""
    mode = metrics.get('mode', 'unknown')
    if metrics.get('n', 0) == 0:
        return f'No predictions to score (predictor={predictor_name}).'

    lines = []
    lines.append('=' * 80)
    lines.append(f'META-REVIEWER EVALUATION  (predictor: {predictor_name}, mode: {mode})')
    lines.append('=' * 80)
    lines.append(f"N rows total: {metrics['n']}")

    if mode == 'axis':
        lines.append('')
        lines.append('Per-axis LABEL accuracy (scored on rows with consensus GT):')
        for axis in ('correctness', 'significance', 'evidence'):
            s = metrics.get('label_accuracy_per_axis', {}).get(axis, {})
            if s.get('accuracy') is None:
                lines.append(f"  {axis:<14} N/A (no scorable rows)")
            else:
                lines.append(
                    f"  {axis:<14} {s['accuracy']:.3f}  "
                    f"({s['n_correct']}/{s['n_scored']})"
                )

        reduced = metrics.get('reduced_6class') or {}
        if reduced.get('accuracy') is not None:
            lines.append('')
            lines.append('Reduced 6-class accuracy (GT restricted to classes reachable by axis mode):')
            lines.append(f"  N scored: {reduced['n_scored']}")
            lines.append(f"  accuracy: {reduced['accuracy']:.3f}")
            lines.append(f"  macro F1: {reduced.get('macro_f1', float('nan')):.3f}")
            lines.append(f"  weighted F1: {reduced.get('weighted_f1', float('nan')):.3f}")
            per_class = reduced.get('per_class_report', {})
            if per_class:
                lines.append('')
                lines.append(f"  {'class':<45} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
                for name, vals in per_class.items():
                    if name in ('accuracy', 'macro avg', 'weighted avg'):
                        continue
                    if not isinstance(vals, dict):
                        continue
                    lines.append(
                        f"  {name[:44]:<45} {vals['precision']:>10.3f} "
                        f"{vals['recall']:>10.3f} {vals['f1-score']:>10.3f} "
                        f"{int(vals['support']):>10}"
                    )

    elif mode == 'tenclass':
        acc = metrics.get('overall_accuracy')
        if acc is None:
            lines.append('No rows had a derivable 10-class ground truth.')
        else:
            lines.append(f"N scored: {metrics.get('n_scored', 0)} "
                         f"(unscorable: {metrics.get('n_unscorable', 0)})")
            lines.append(f"10-class accuracy: {acc:.3f}")
            lines.append(f"Macro F1:          {metrics['macro_f1']:.3f}")
            lines.append(f"Weighted F1:       {metrics['weighted_f1']:.3f}")
            lines.append('')
            lines.append('Per-class precision / recall / F1 (on classes actually observed):')
            lines.append(f"  {'class':<45} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
            for name, vals in metrics.get('per_class_report', {}).items():
                if name in ('accuracy', 'macro avg', 'weighted avg'):
                    continue
                if not isinstance(vals, dict):
                    continue
                lines.append(
                    f"  {name[:44]:<45} {vals['precision']:>10.3f} "
                    f"{vals['recall']:>10.3f} {vals['f1-score']:>10.3f} "
                    f"{int(vals['support']):>10}"
                )

    lines.append('')
    return '\n'.join(lines)
