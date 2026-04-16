"""
Full-similarity compute (LLM-JUDGE path): run the same within-paper pair
set as compute_full_similarity_embedding.py through a thinking-mode
frontier LLM as a 4-way classifier (same prompt/taxonomy as
expert_annotation_similarity/baselines/llm_classifier.py).

Main research question (same as the embedding path):
    "How similar are AI reviews to human reviews?"

Why a separate script?
    Embeddings give you a continuous cosine score per pair — great for
    distributions, nearest-neighbor analyses, and ranking AI models. But a
    thinking-mode LLM judge is sharper at the fine-grained 4-way taxonomy
    (near-paraphrase vs convergent vs topical neighbor vs unrelated) and
    gave +15pp binary accuracy over Azure embedding on the 164-pair curated
    eval. Here we apply the same judge to every within-paper pair so the
    paper can cite a "judged" number alongside the cosine number.

Scale warning:
    The expert_annotation HF config yields ~66,400 within-paper pairs
    (~35k H-H, ~6k A-A, ~25k H-A). At 3 judges × 66k × ~10s per
    thinking-mode call, the serial cost is >500 hours. You MUST use
    --concurrency; a pool of 16 reduces the wall-clock per judge to
    roughly 8-12 hours. The script is **resumable**: each result is
    flushed to the JSONL file on completion, and a rerun skips any pair
    whose (paper_id, reviewer_a, item_a_num, reviewer_b, item_b_num)
    key (sorted to be order-invariant) already appears in the file. If
    the process is killed you can restart the same command and it will
    pick up where it left off.

Outputs in `--output-dir`:
    pairs_llm_<model-slug>.jsonl      — one JSON object per pair
    metadata_llm_<model-slug>.json    — model, counts, timings

Per-pair JSONL record (minimal, to keep the 66k-line file manageable):
    {
      "paper_id": 1,
      "pair_type": "H-A",            // H-H | A-A | H-A
      "same_reviewer": false,
      "item_a": {"reviewer_id": "...", "reviewer_type": "Human", "review_item_number": 3},
      "item_b": {"reviewer_id": "...", "reviewer_type": "AI",    "review_item_number": 7},
      "parsed_answer": "same subject, same argument, different evidence",
      "parsed_binary": "similar",
      "elapsed_seconds": 14.2,
      "error": null
    }

Full response / reasoning_content is NOT written by default (it would
balloon the output file to ~130 MB per model). Pass `--save-reasoning`
to include it if you want to audit specific pairs.

Text-only mode only: no images, no code artifacts. At 66k pairs × 3
models, paying the 2 GB `submitted_papers` cost plus per-pair image
loading is a non-starter. The 4-way taxonomy is a text-comparison
task and the `expert_annotation_similarity` ablation showed the
text-only judge lands within ~1pp of the multimodal judge on the 164
curated pairs.

Usage:
    # run the judge on all pairs for one model (resume-safe)
    python compute_full_similarity_llm.py \
      --model litellm_proxy/anthropic/claude-opus-4-6 \
      --concurrency 16

    # smoke-test on 3 papers
    python compute_full_similarity_llm.py \
      --model litellm_proxy/azure_ai/gpt-5.4 --limit 3 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# HF Hub network timeouts — apply BEFORE importing huggingface_hub.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

# --- Path plumbing ---------------------------------------------------------
_HERE = Path(__file__).resolve().parent                     # .../similarity_check/full_similarity
_SIMCHECK_DIR = _HERE.parent                                 # .../similarity_check
_BENCH_DIR = _SIMCHECK_DIR.parent                            # .../peerreview_bench
_EA_DIR = _SIMCHECK_DIR / 'expert_annotation_similarity'     # reuse prompts + retry
_EA_BASELINES = _EA_DIR / 'baselines'
_META_REVIEW_DIR = _BENCH_DIR / 'meta_review'

for p in (_META_REVIEW_DIR, _BENCH_DIR, _EA_DIR, _EA_BASELINES, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the expert_annotation_similarity prompt + helpers verbatim so the
# judge here returns labels in exactly the same format as the 164-pair
# baseline.
from prompts import (  # noqa: E402
    FOURWAY_SYSTEM_PROMPT,
    FOURWAY_USER_PROMPT_TEMPLATE,
    fourway_to_binary,
)
# llm_classifier owns the retry+thinking-mode logic (`_call_llm_with_reasoning`),
# the answer extractor (`extract_4way_answer`), and the model/slug helpers.
from llm_classifier import (  # noqa: E402
    _call_llm_with_reasoning,
    _slug,
    bare_name_from_model,
    build_reasoning_kwargs,
    extract_4way_answer,
)
from model_config import get_max_output_tokens  # noqa: E402

from load_data import load_expert_annotation_rows  # noqa: E402


# ---------------------------------------------------------------------------
# Item prep — identical to compute_full_similarity_embedding.py
# ---------------------------------------------------------------------------

def _dedupe_items(rows: List[Dict]) -> List[Dict]:
    seen: Dict[Tuple[int, str, int], Dict] = {}
    for r in rows:
        key = (int(r['paper_id']), r['reviewer_id'], int(r['review_item_number']))
        if key not in seen:
            seen[key] = r
    return list(seen.values())


def _group_by_paper(items: List[Dict]) -> Dict[int, Dict[str, List[int]]]:
    groups: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: {'H': [], 'A': []})
    for idx, item in enumerate(items):
        pid = int(item['paper_id'])
        tp = 'A' if item['reviewer_type'] == 'AI' else 'H'
        groups[pid][tp].append(idx)
    return dict(groups)


def _item_key_dict(item: Dict) -> Dict:
    return {
        'reviewer_id': item['reviewer_id'],
        'reviewer_type': item['reviewer_type'],
        'review_item_number': int(item['review_item_number']),
    }


def _pair_id_tuple(paper_id: int, a: Dict, b: Dict) -> Tuple:
    """Order-invariant stable pair key used to detect already-scored pairs
    across resumes. Two items' identity triples are sorted so that (A, B)
    and (B, A) collapse to the same tuple."""
    ka = (a['reviewer_id'], int(a['review_item_number']))
    kb = (b['reviewer_id'], int(b['review_item_number']))
    ka, kb = tuple(sorted([ka, kb]))
    return (int(paper_id), ka[0], ka[1], kb[0], kb[1])


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def _generate_pairs(
    items: List[Dict],
    groups: Dict[int, Dict[str, List[int]]],
) -> List[Dict[str, Any]]:
    """Produce every within-paper pair in a stable order matching the
    embedding path (H-H upper triangle, then A-A upper triangle, then H-A
    full grid, paper_ids ascending)."""
    pairs: List[Dict[str, Any]] = []
    for pid in sorted(groups.keys()):
        g = groups[pid]
        H = g['H']
        A = g['A']

        for i in range(len(H)):
            for j in range(i + 1, len(H)):
                ia, ib = H[i], H[j]
                pairs.append({
                    'paper_id': pid,
                    'pair_type': 'H-H',
                    'same_reviewer': items[ia]['reviewer_id'] == items[ib]['reviewer_id'],
                    'idx_a': ia, 'idx_b': ib,
                })

        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                ia, ib = A[i], A[j]
                pairs.append({
                    'paper_id': pid,
                    'pair_type': 'A-A',
                    'same_reviewer': items[ia]['reviewer_id'] == items[ib]['reviewer_id'],
                    'idx_a': ia, 'idx_b': ib,
                })

        for i in H:
            for j in A:
                pairs.append({
                    'paper_id': pid,
                    'pair_type': 'H-A',
                    'same_reviewer': False,
                    'idx_a': i, 'idx_b': j,
                })
    return pairs


# ---------------------------------------------------------------------------
# LLM prompt building (text-only)
# ---------------------------------------------------------------------------

def _build_messages(paper_content: str, item_a: Dict, item_b: Dict) -> List[Dict[str, Any]]:
    user_text = FOURWAY_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_content,
        reviewer_a=item_a['reviewer_id'],
        reviewer_b=item_b['reviewer_id'],
        item_a=item_a['review_item'],
        item_b=item_b['review_item'],
    )
    return [
        {'role': 'system', 'content': FOURWAY_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_text},
    ]


# ---------------------------------------------------------------------------
# Resume: load already-scored pair ids from existing JSONL
# ---------------------------------------------------------------------------

def _load_existing_pair_ids(pairs_path: Path) -> set:
    """Read an existing pairs JSONL (from a previous run) and return the
    set of order-invariant pair id tuples already scored. Silently tolerates
    a missing file, blank lines, and partially-written final lines."""
    seen: set = set()
    if not pairs_path.exists():
        return seen
    with pairs_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Truncated final line from a crash — drop it.
                print(f'  note: dropping truncated line {line_num} in {pairs_path.name}',
                      file=sys.stderr)
                continue
            try:
                seen.add(_pair_id_tuple(rec['paper_id'], rec['item_a'], rec['item_b']))
            except KeyError:
                # Legacy record missing expected keys — skip.
                continue
    return seen


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    model: str,
    output_dir: Path,
    limit: Optional[int],
    *,
    max_tokens: Optional[int],
    temperature: float,
    concurrency: int,
    save_reasoning: bool,
) -> None:
    print('Loading expert_annotation rows from HuggingFace...')
    rows = load_expert_annotation_rows()
    print(f'  {len(rows)} raw rows')

    items = _dedupe_items(rows)
    items = [r for r in items if (r.get('review_item') or '').strip()]
    print(f'  {len(items)} unique non-empty items after dedupe')

    if limit is not None:
        all_pids = sorted({int(r['paper_id']) for r in items})
        keep = set(all_pids[:limit])
        items = [r for r in items if int(r['paper_id']) in keep]
        print(f'  --limit {limit} papers: {len(items)} items retained')

    # Build a per-paper → paper_content map so workers don't keep repeating the
    # same string lookup. The expert_annotation schema has paper_content on
    # every row, so we pick one per paper.
    paper_content_by_id: Dict[int, str] = {}
    for r in items:
        pid = int(r['paper_id'])
        if pid not in paper_content_by_id:
            paper_content_by_id[pid] = r.get('paper_content') or ''

    groups = _group_by_paper(items)
    n_papers = len(groups)
    h_count = sum(len(g['H']) for g in groups.values())
    a_count = sum(len(g['A']) for g in groups.values())
    print(f'  {n_papers} papers, {h_count} Human items, {a_count} AI items')

    pairs = _generate_pairs(items, groups)
    n_total = len(pairs)
    print(f'  {n_total} total within-paper pairs to score')

    # ---- Model setup ------------------------------------------------------
    bare_model = bare_name_from_model(model)
    effective_max_tokens = (
        max_tokens if max_tokens is not None else get_max_output_tokens(bare_model)
    )
    reasoning_kwargs = build_reasoning_kwargs(model, effective_max_tokens)

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(model)
    pairs_path = output_dir / f'pairs_llm_{slug}.jsonl'
    meta_path = output_dir / f'metadata_llm_{slug}.json'

    print(f'\nModel:             {model}')
    print(f'Max output tokens: {effective_max_tokens}')
    print(f'Temperature:       {temperature}')
    print(f'Reasoning kwargs:  {reasoning_kwargs}')
    print(f'Concurrency:       {concurrency} worker(s)')
    print(f'Pairs file:        {pairs_path}')
    print(f'Metadata file:     {meta_path}')

    # ---- Resume: skip already-scored pairs --------------------------------
    print('\nChecking for existing results to resume from...')
    already = _load_existing_pair_ids(pairs_path)
    if already:
        print(f'  found {len(already)} already-scored pairs; will skip')
    todo: List[Dict[str, Any]] = []
    for p in pairs:
        a = items[p['idx_a']]
        b = items[p['idx_b']]
        pid = _pair_id_tuple(p['paper_id'], a, b)
        if pid in already:
            continue
        todo.append(p)
    n_todo = len(todo)
    n_skipped = n_total - n_todo
    print(f'  resuming: {n_skipped} done, {n_todo} remaining (of {n_total} total)')

    if not todo:
        print('\nNothing to do — all pairs already scored. Exiting.')
        _write_metadata(meta_path, model, effective_max_tokens, temperature,
                        reasoning_kwargs, n_papers, h_count, a_count,
                        n_total, n_todo=0, n_completed=n_total, n_errors=0,
                        n_parsed=0, started_at=None)
        return

    # Progress tracking (cumulative — include already-done in the starting count)
    n_completed = n_skipped
    n_errors = 0
    n_parsed = 0
    lock = threading.Lock()

    # Open pairs file in append mode; create parent dir if needed.
    pairs_fh = pairs_path.open('a', buffering=1)  # line-buffered so crashes lose ≤1 line

    pbar = tqdm(
        total=n_total,
        initial=n_skipped,
        desc=bare_model,
        unit='pair',
        dynamic_ncols=True,
        mininterval=1.0,
    )

    started_at = time.time()

    def _score_one(pair: Dict[str, Any]) -> Dict[str, Any]:
        """Per-pair worker."""
        idx_a = pair['idx_a']
        idx_b = pair['idx_b']
        item_a = items[idx_a]
        item_b = items[idx_b]
        pid = int(pair['paper_id'])
        paper_content = paper_content_by_id.get(pid, '')

        # For H-A pairs, put the Human first for readability (matches the
        # embedding path convention).
        if pair['pair_type'] == 'H-A' and item_a['reviewer_type'] == 'AI':
            item_a, item_b = item_b, item_a

        messages = _build_messages(paper_content, item_a, item_b)

        t0 = time.time()
        response_text = ''
        reasoning_content: Optional[str] = None
        error: Optional[str] = None
        try:
            call_result = _call_llm_with_reasoning(
                model=model,
                messages=messages,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                extra_kwargs=reasoning_kwargs,
            )
            response_text = call_result['content']
            reasoning_content = call_result['reasoning_content']
        except Exception as e:
            error = f'{type(e).__name__}: {e}'
        elapsed = time.time() - t0

        parsed = extract_4way_answer(response_text) if response_text else None
        parsed_binary: Optional[str] = fourway_to_binary(parsed) if parsed is not None else None

        rec: Dict[str, Any] = {
            'paper_id': pid,
            'pair_type': pair['pair_type'],
            'same_reviewer': bool(pair['same_reviewer']),
            'item_a': _item_key_dict(item_a),
            'item_b': _item_key_dict(item_b),
            'parsed_answer': parsed,
            'parsed_binary': parsed_binary,
            'elapsed_seconds': round(elapsed, 3),
            'error': error,
        }
        if save_reasoning:
            rec['response'] = response_text
            rec['reasoning_content'] = reasoning_content
        return rec

    def _on_result(rec: Dict[str, Any]) -> None:
        nonlocal n_completed, n_errors, n_parsed
        with lock:
            pairs_fh.write(json.dumps(rec) + '\n')
            pairs_fh.flush()
            n_completed += 1
            if rec['parsed_answer'] is not None:
                n_parsed += 1
            if rec['error']:
                n_errors += 1
                tqdm.write(f"  [pair {n_completed}/{n_total}] ERROR: {rec['error']}")
            pbar.set_postfix({
                'parsed': f'{n_parsed}/{n_completed - n_skipped}',
                'err': n_errors,
            })
            pbar.update(1)

    def _worker(pair: Dict[str, Any]) -> None:
        try:
            rec = _score_one(pair)
        except Exception as e:
            # Defensive — _score_one should catch its own errors, but
            # don't let a worker exception kill the whole run.
            a = items[pair['idx_a']]
            b = items[pair['idx_b']]
            rec = {
                'paper_id': int(pair['paper_id']),
                'pair_type': pair['pair_type'],
                'same_reviewer': bool(pair['same_reviewer']),
                'item_a': _item_key_dict(a),
                'item_b': _item_key_dict(b),
                'parsed_answer': None,
                'parsed_binary': None,
                'elapsed_seconds': 0.0,
                'error': f'worker_unhandled: {type(e).__name__}: {e}',
            }
        _on_result(rec)

    try:
        if concurrency == 1:
            for pair in todo:
                _worker(pair)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_worker, pair) for pair in todo]
                for _ in as_completed(futures):
                    pass
    finally:
        pbar.close()
        pairs_fh.close()

    elapsed_total = time.time() - started_at
    print(
        f'\nFinished. Scored {n_completed - n_skipped} new pairs '
        f'({n_parsed} parsed, {n_errors} errors) in {elapsed_total:.1f}s.  '
        f'Total in file: {n_completed}/{n_total}.'
    )

    _write_metadata(
        meta_path, model, effective_max_tokens, temperature, reasoning_kwargs,
        n_papers, h_count, a_count, n_total,
        n_todo=n_todo, n_completed=n_completed, n_errors=n_errors,
        n_parsed=n_parsed, started_at=started_at,
    )
    print(f'Wrote metadata: {meta_path}')


def _write_metadata(
    meta_path: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    reasoning_kwargs: Dict[str, Any],
    n_papers: int,
    h_count: int,
    a_count: int,
    n_total: int,
    *,
    n_todo: int,
    n_completed: int,
    n_errors: int,
    n_parsed: int,
    started_at: Optional[float],
) -> None:
    meta: Dict[str, Any] = {
        'mode': 'full_similarity_llm_4way',
        'model': model,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'reasoning_kwargs': reasoning_kwargs,
        'n_papers': n_papers,
        'n_human_items': h_count,
        'n_ai_items': a_count,
        'n_total_pairs_expected': n_total,
        'n_pairs_scored_in_file': n_completed,
        'n_parsed': n_parsed,
        'n_errors': n_errors,
        'n_in_this_run': n_todo,
    }
    if started_at is not None:
        meta['last_run_seconds'] = round(time.time() - started_at, 1)
    meta_path.write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description='Run an LLM-as-judge (4-way) over every within-paper pair '
                    'in the expert_annotation HF config. Resumable.')
    parser.add_argument(
        '--model', type=str, required=True,
        help='LiteLLM model id, with or without the `litellm_proxy/` prefix. '
             'E.g. litellm_proxy/azure_ai/gpt-5.4, '
             'litellm_proxy/gemini/gemini-3.1-pro-preview, '
             'litellm_proxy/anthropic/claude-opus-4-6.',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=_BENCH_DIR / 'outputs' / 'full_similarity',
        help='Where to write pairs_llm_*.jsonl and metadata_llm_*.json',
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Smoke test: only the first N papers.')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Override completion max_tokens. Default: the '
                             "model's catalog max_output_tokens.")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature. Default 1.0 — required by '
                             'Anthropic extended thinking, recommended for '
                             'Gemini 3 reasoning.')
    parser.add_argument('--concurrency', type=int, default=16,
                        help='Concurrent LLM workers. Default 16. Each worker '
                             'has its own rate-limit retry; 32-64 are safe to '
                             'try if the proxy has headroom.')
    parser.add_argument('--save-reasoning', action='store_true',
                        help='Also save the full response text and '
                             "thinking-mode reasoning_content in each JSONL "
                             'record. Off by default — turning it on balloons '
                             'the output file to >100 MB per model.')
    args = parser.parse_args()

    run(
        model=args.model,
        output_dir=args.output_dir,
        limit=args.limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        save_reasoning=args.save_reasoning,
    )


if __name__ == '__main__':
    main()
