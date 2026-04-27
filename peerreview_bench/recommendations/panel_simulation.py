#!/usr/bin/env python3
"""
Simulate different reviewer panel compositions and measure quality metrics.

Baselines:
  1. 3 Human Reviewers
  2. 2 Human Reviewers + 1 AI Reviewer
  3. 1 Human Reviewer + 2 AI Reviewers
  4. 3 AI Reviewers
  5-8. Same as 1-4 but filtered through AI Meta-Reviewer (keep only FG items)

Metrics per panel:
  - # Total review items
  - # Fully positive items (expert-judged: Correct + Significant + Sufficient)
  - % Fully positive
  - # Unique items (no similar item from any other reviewer in the panel)
  - % Unique items
  - # Fully positive + unique items
  - % Fully positive + unique items

All metrics computed per paper, then averaged with 95% bootstrap CIs.

Usage:
    python3 panel_simulation.py
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from itertools import combinations
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_FORCE_REDOWNLOAD', '0')

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent
sys.path.insert(0, str(_BENCH))

from load_data import load_annotations, load_meta_reviewer

NORM = {
    'claude-opus-4-5': 'Claude', 'review_claude-opus-4-7': 'Claude',
    'gpt-5.2': 'GPT',
    'gemini-3.0-pro-preview': 'Gemini', 'review_gemini-3.1-pro-preview': 'Gemini',
    'Human_1': 'Human_1', 'Human_2': 'Human_2', 'Human_3': 'Human_3',
}

SIMILARITY_FILE = _BENCH / 'outputs' / 'full_similarity' / 'pairs_llm_azure_ai__gpt-5_4.jsonl'
DROPPED_PAPERS = {11, 20, 22}
N_BOOT = 10_000
SEED = 42

ORDINAL = {
    'same subject, same argument, same evidence': 3,
    'same subject, same argument, different evidence': 2,
    'same subject, different argument': 1,
    'different subject': 0,
}

AI_REVIEWERS = ['GPT', 'Claude', 'Gemini']
HUMAN_REVIEWERS = ['Human_1', 'Human_2', 'Human_3']


def load_expert_items():
    """Load primary expert annotations, return per-paper dict of items."""
    items, _ = load_annotations('primary')
    by_paper = defaultdict(list)
    for it in items:
        if it.paper_id in DROPPED_PAPERS:
            continue
        is_fg = (it.correctness == 'Correct' and
                 it.significance_numeric == 2 and
                 it.evidence_numeric == 1)
        by_paper[it.paper_id].append({
            'paper_id': it.paper_id,
            'reviewer_id': it.reviewer_id,
            'item_number': it.item_number,
            'reviewer_type': it.reviewer_type,
            'is_fg': is_fg,
        })
    return dict(by_paper)


def load_similarity_index():
    """Build similarity index from full similarity JSONL.
    Returns: (paper_id, reviewer_a, item_a, reviewer_b) -> max ordinal."""
    sim = defaultdict(int)
    with open(SIMILARITY_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r.get('paper_id') in DROPPED_PAPERS:
                continue
            if r.get('error') is not None and str(r['error']) != 'None':
                continue
            # Only inter-reviewer pairs (not self-similarity)
            if r.get('same_reviewer', False):
                continue
            ov = ORDINAL.get(r['parsed_answer'], 0)
            a, b = r['item_a'], r['item_b']
            key_ab = (r['paper_id'], a['reviewer_id'], a['review_item_number'], b['reviewer_id'])
            sim[key_ab] = max(sim[key_ab], ov)
            key_ba = (r['paper_id'], b['reviewer_id'], b['review_item_number'], a['reviewer_id'])
            sim[key_ba] = max(sim[key_ba], ov)
    return dict(sim)


def _load_metareview_dir(results_dir, predictions):
    """Load meta-reviewer predictions from a directory of *_metareview.json files."""
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
                meta_fg = (item.get('correctness') == 'Correct' and
                           item.get('significance') == 'Significant' and
                           item.get('evidence') == 'Sufficient')
                predictions[(pid, rid, item['item_number'])] = meta_fg


def _load_precision_eval(predictions):
    """Load AI meta-reviewer predictions from precision evaluation results.

    Maps model-specific precision files to normalized AI family names:
      precision_gpt-5.2.json        -> GPT
      precision_claude-opus-4-5.json -> Claude
      precision_gemini-3.0-pro-preview.json -> Gemini
    """
    eval_dir = _BENCH / 'outputs' / 'eval'
    family_map = {
        'precision_gpt-5.2.json': 'GPT',
        'precision_claude-opus-4-5.json': 'Claude',
        'precision_gemini-3.0-pro-preview.json': 'Gemini',
    }
    for fname, family in family_map.items():
        fpath = eval_dir / fname
        if not fpath.exists():
            continue
        data = json.load(open(fpath))
        for item in data['per_item']:
            meta_fg = item.get('is_fully_good', False)
            predictions[(item['paper_id'], family, item['item_number'])] = meta_fg


def load_meta_reviewer_predictions():
    """Load GPT-5.4 meta-reviewer predictions from all sources.

    Sources:
      1. metareview_bench results (27 papers, all reviewers)
      2. human_metareview_results (54 gap papers, human reviewers only)
      3. precision evaluation results (81 papers, AI reviewers only)

    Returns: predictions dict of (pid, rid, item#) -> is_fg
    """
    predictions = {}

    # 1. Meta-review bench (27 papers — has human + AI reviewer predictions)
    bench_dir = _BENCH / 'metareview_bench' / 'results' / 'litellm_proxy__azure_ai__gpt_5_4'
    _load_metareview_dir(bench_dir, predictions)

    # 2. Human meta-review results (54 gap papers — human reviewers only)
    human_dir = _HERE / 'human_metareview_results' / 'openai__gpt_5_4'
    if human_dir.exists():
        _load_metareview_dir(human_dir, predictions)

    # 3. Precision eval results (81 papers — AI reviewers only)
    _load_precision_eval(predictions)

    return predictions


def is_unique(paper_id, reviewer_id, item_number, panel_reviewers, sim_index, threshold=2):
    """Check if an item has no similar counterpart from any other reviewer in the panel."""
    for other_rev in panel_reviewers:
        if other_rev == reviewer_id:
            continue
        max_sim = sim_index.get((paper_id, reviewer_id, item_number, other_rev), 0)
        if max_sim >= threshold:
            return False
    return True


def compute_panel_metrics(paper_items, panel_reviewers, sim_index, meta_preds=None):
    """Compute metrics for a panel on one paper.

    If meta_preds is provided, filter to only meta-reviewer-approved items.
    Returns dict of metrics or None if panel has no items on this paper.
    """
    # Collect items from panel reviewers
    items = [it for it in paper_items if it['reviewer_id'] in panel_reviewers]

    if not items:
        return None

    # Apply meta-reviewer filter if provided
    if meta_preds is not None:
        filtered = []
        for it in items:
            key = (it['paper_id'], it['reviewer_id'], it['item_number'])
            if meta_preds.get(key, False):  # keep only meta-FG items
                filtered.append(it)
        items = filtered
        if not items:
            return {
                'n_total': 0, 'n_fg': 0, 'n_not_fg': 0, 'pct_fg': 0,
                'n_unique': 0, 'pct_unique': 0,
                'n_fg_unique': 0, 'pct_fg_unique': 0,
            }

    n_total = len(items)
    n_fg = sum(1 for it in items if it['is_fg'])

    # Uniqueness
    n_unique = 0
    n_fg_unique = 0
    for it in items:
        if is_unique(it['paper_id'], it['reviewer_id'], it['item_number'],
                     panel_reviewers, sim_index):
            n_unique += 1
            if it['is_fg']:
                n_fg_unique += 1

    return {
        'n_total': n_total,
        'n_fg': n_fg,
        'n_not_fg': n_total - n_fg,
        'pct_fg': n_fg / n_total if n_total > 0 else 0,
        'n_unique': n_unique,
        'pct_unique': n_unique / n_total if n_total > 0 else 0,
        'n_fg_unique': n_fg_unique,
        'pct_fg_unique': n_fg_unique / n_total if n_total > 0 else 0,
    }


def bootstrap_ci(paper_metrics, n_boot=N_BOOT, seed=SEED):
    """Paper-level bootstrap CI for each metric."""
    rng = np.random.RandomState(seed)
    paper_ids = list(paper_metrics.keys())
    n = len(paper_ids)

    metric_keys = ['n_total', 'n_fg', 'n_not_fg', 'pct_fg', 'n_unique', 'pct_unique',
                   'n_fg_unique', 'pct_fg_unique']

    # Observed means
    obs = {}
    for mk in metric_keys:
        obs[mk] = np.mean([paper_metrics[pid][mk] for pid in paper_ids])

    # Bootstrap
    boot = {mk: np.empty(n_boot) for mk in metric_keys}
    for b in range(n_boot):
        sampled = rng.choice(paper_ids, size=n, replace=True)
        for mk in metric_keys:
            boot[mk][b] = np.mean([paper_metrics[pid][mk] for pid in sampled])

    results = {}
    for mk in metric_keys:
        lo = np.percentile(boot[mk], 2.5)
        hi = np.percentile(boot[mk], 97.5)
        results[mk] = (obs[mk], lo, hi)

    return results


def run_baseline(name, paper_items_dict, sim_index, panel_fn, meta_preds=None,
                 restrict_papers=None):
    """Run a baseline across all papers.

    panel_fn: function(paper_id, available_reviewers) -> list of panel reviewer sets
              Returns multiple panels if averaging over combinations.
    """
    paper_metrics = {}

    for pid, items in paper_items_dict.items():
        if restrict_papers and pid not in restrict_papers:
            continue

        available = set(it['reviewer_id'] for it in items)
        panels = panel_fn(pid, available)

        if not panels:
            continue

        # Average metrics across all valid panels for this paper
        all_panel_metrics = []
        for panel_revs in panels:
            if not all(r in available for r in panel_revs):
                continue
            m = compute_panel_metrics(items, panel_revs, sim_index, meta_preds)
            if m is not None:
                all_panel_metrics.append(m)

        if all_panel_metrics:
            avg = {}
            for mk in all_panel_metrics[0]:
                avg[mk] = np.mean([m[mk] for m in all_panel_metrics])
            paper_metrics[pid] = avg

    if not paper_metrics:
        return None

    results = bootstrap_ci(paper_metrics)
    results['n_papers'] = len(paper_metrics)
    results['name'] = name
    return results


def main():
    print("Loading data...")
    paper_items = load_expert_items()
    print(f"  Papers: {len(paper_items)}")

    print("Loading similarity index...")
    sim_index = load_similarity_index()
    print(f"  Pairs: {len(sim_index)}")

    print("Loading meta-reviewer predictions...")
    meta_preds = load_meta_reviewer_predictions()
    meta_papers = set(k[0] for k in meta_preds.keys())
    print(f"  Items: {len(meta_preds)}, Papers: {len(meta_papers)}")

    # Common paper set: papers with 3 humans + 3 AIs + meta-reviewer preds for all
    common_papers = set()
    for pid, items in paper_items.items():
        revs = set(it['reviewer_id'] for it in items)
        has_3h = sum(1 for r in HUMAN_REVIEWERS if r in revs) >= 3
        has_3a = sum(1 for r in AI_REVIEWERS if r in revs) >= 3
        has_meta = all(
            (pid, r, it['item_number']) in meta_preds
            for it in items
            for r in [it['reviewer_id']]
        )
        if has_3h and has_3a and has_meta:
            common_papers.add(pid)
    print(f"  Common paper set: {len(common_papers)}")

    # Define panel functions
    def panel_3h(pid, avail):
        humans = [r for r in HUMAN_REVIEWERS if r in avail]
        if len(humans) >= 3:
            return [humans[:3]]
        return []

    def panel_2h1a(pid, avail):
        humans = [r for r in HUMAN_REVIEWERS if r in avail]
        ais = [r for r in AI_REVIEWERS if r in avail]
        panels = []
        for h_combo in combinations(humans, min(2, len(humans))):
            if len(h_combo) < 2:
                continue
            for ai in ais:
                panels.append(list(h_combo) + [ai])
        return panels

    def panel_1h2a(pid, avail):
        humans = [r for r in HUMAN_REVIEWERS if r in avail]
        ais = [r for r in AI_REVIEWERS if r in avail]
        panels = []
        for h in humans:
            for a_combo in combinations(ais, min(2, len(ais))):
                if len(a_combo) < 2:
                    continue
                panels.append([h] + list(a_combo))
        return panels

    def panel_3a(pid, avail):
        ais = [r for r in AI_REVIEWERS if r in avail]
        if len(ais) >= 3:
            return [ais[:3]]
        return []

    # All 8 rows use the same common paper set for fair comparison
    baselines = [
        ("3 Human", panel_3h, None, common_papers),
        ("2 Human + 1 AI", panel_2h1a, None, common_papers),
        ("1 Human + 2 AI", panel_1h2a, None, common_papers),
        ("3 AI", panel_3a, None, common_papers),
        ("3 Human → Meta-Rev", panel_3h, meta_preds, common_papers),
        ("2H + 1AI → Meta-Rev", panel_2h1a, meta_preds, common_papers),
        ("1H + 2AI → Meta-Rev", panel_1h2a, meta_preds, common_papers),
        ("3 AI → Meta-Rev", panel_3a, meta_preds, common_papers),
    ]

    all_results = []
    for name, panel_fn, mp, restrict in baselines:
        print(f"\n  Running: {name}...")
        result = run_baseline(name, paper_items, sim_index, panel_fn,
                              meta_preds=mp, restrict_papers=restrict)
        if result:
            all_results.append(result)
            obs = result
            print(f"    Papers: {obs['n_papers']}")
            print(f"    Total items: {obs['n_total'][0]:.1f} [{obs['n_total'][1]:.1f}, {obs['n_total'][2]:.1f}]")
            print(f"    FG items: {obs['n_fg'][0]:.1f} [{obs['n_fg'][1]:.1f}, {obs['n_fg'][2]:.1f}]")
            print(f"    % FG: {obs['pct_fg'][0]*100:.1f}% [{obs['pct_fg'][1]*100:.1f}, {obs['pct_fg'][2]*100:.1f}]")
            print(f"    Unique: {obs['n_unique'][0]:.1f} [{obs['n_unique'][1]:.1f}, {obs['n_unique'][2]:.1f}]")
            print(f"    % Unique: {obs['pct_unique'][0]*100:.1f}% [{obs['pct_unique'][1]*100:.1f}, {obs['pct_unique'][2]*100:.1f}]")
            print(f"    FG+Unique: {obs['n_fg_unique'][0]:.1f} [{obs['n_fg_unique'][1]:.1f}, {obs['n_fg_unique'][2]:.1f}]")
            print(f"    % FG+Unique: {obs['pct_fg_unique'][0]*100:.1f}% [{obs['pct_fg_unique'][1]*100:.1f}, {obs['pct_fg_unique'][2]*100:.1f}]")

    # Generate LaTeX table
    generate_latex(all_results)


def fmt_count_ci(obs, lo, hi):
    return f"{obs:.1f}" + r"{\scriptsize\,[" + f"{lo:.1f}, {hi:.1f}" + r"]}"

def fmt_pct_ci(obs, lo, hi):
    return f"{obs*100:.1f}" + r"\%" + r"{\scriptsize\,[" + f"{lo*100:.1f}, {hi*100:.1f}" + r"]}"


def generate_latex(results):
    # Panel composition mapping: name -> (n_human, n_ai, meta)
    panel_map = {
        '3 Human':              ('3', '0', r'\xmark'),
        '2 Human + 1 AI':       ('2', '1', r'\xmark'),
        '1 Human + 2 AI':       ('1', '2', r'\xmark'),
        '3 AI':                 ('0', '3', r'\xmark'),
        '3 Human → Meta-Rev':   ('3', '0', r'\cmark'),
        '2H + 1AI → Meta-Rev':  ('2', '1', r'\cmark'),
        '1H + 2AI → Meta-Rev':  ('1', '2', r'\cmark'),
        '3 AI → Meta-Rev':      ('0', '3', r'\cmark'),
    }

    lines = [
        r'\begin{table*}[t]',
        r'    \centering',
        r'    \fontsize{7}{9}\selectfont',
        r'    \setlength{\tabcolsep}{6pt}',
        r'    \renewcommand{\arraystretch}{1.0}',
        r'    \begin{tabular}{ccc cccc}',
        r'    \toprule',
        r'    \multicolumn{3}{c}{\textbf{Panel Composition}} & & \multicolumn{1}{c}{\textbf{Diversity}} & \multicolumn{1}{c}{\textbf{Quality}} & \multicolumn{1}{c}{\textbf{Diversity \& Quality}} \\',
        r'    \cmidrule(lr){1-3} \cmidrule(lr){5-5} \cmidrule(lr){6-6} \cmidrule(lr){7-7}',
        r'    \makecell[c]{\textbf{\# Human} \\ \textbf{Reviewers}} & \makecell[c]{\textbf{\# AI} \\ \textbf{Reviewers}} & \makecell[c]{\textbf{+ AI Meta-Reviewer} \\ \textbf{Filtering}} & \textbf{\# Total Items} & \makecell[c]{\textbf{\# Unique Items} \\ ($\uparrow$ better)} & \makecell[c]{\textbf{\# Not Fully} \\ \textbf{Positive Items} ($\downarrow$ better)} & \makecell[c]{\textbf{\# Fully Positive+} \\ \textbf{Unique Items} ($\uparrow$ better)} \\',
        r'    \midrule',
    ]

    for i, r in enumerate(results):
        if i == 4:
            lines.append(r'    \midrule')

        n_human, n_ai, meta = panel_map[r['name']]
        n_tot = fmt_count_ci(*r['n_total'])
        n_uniq = fmt_count_ci(*r['n_unique'])
        n_not_fg = fmt_count_ci(*r['n_not_fg'])
        n_fg_uniq = fmt_count_ci(*r['n_fg_unique'])

        lines.append(
            f'    {n_human} & {n_ai} & {meta}'
            f' & {n_tot} & {n_uniq} & {n_not_fg} & {n_fg_uniq} \\\\'
        )

    n_papers = results[0]['n_papers']

    lines.extend([
        r'    \bottomrule',
        r'    \end{tabular}',
        r"    \caption{\textbf{Reviewer panel composition analysis.}",
        r'    For each panel configuration, we report the mean per-paper count of:',
        r'    (i) total items the panel produced,',
        r'    (ii) unique items (no similar counterpart from any other reviewer in the panel, ordinal $<2$) -- a \emph{diversity} measure,',
        r'    (iii) items that are \emph{not} fully positive (the burden a reader or editor must triage) -- a \emph{quality} measure (lower is better),',
        r'    and (iv) items that are simultaneously fully positive and unique (FG+Unique; correct, significant, evidence-sufficient by expert annotation \emph{and} unique to that reviewer) -- a joint \emph{diversity-and-quality} measure.',
        r"    Rows~5--8 apply an AI meta-reviewer (GPT-5.4) filter that removes items the meta-reviewer judges not fully positive, simulating automated quality control.",
        r"    ``2 Human + 1 AI'' averages over all $\binom{3}{2} \times 3 = 9$ possible panels;",
        r"    ``1 Human + 2 AI'' averages over all $3 \times \binom{3}{2} = 9$ panels.",
        f'    All rows are computed on the same {n_papers} papers with complete data (3 human reviewers, 3 AI reviewers, and meta-reviewer predictions for every item).',
        r'    95\% CIs are paper-level bootstrap (10{,}000 resamples).}',
        r'    \label{tab:panel-simulation}',
        r'    \end{table*}',
    ])

    tex = '\n'.join(lines)
    out_path = _HERE / 'table_panel_simulation.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
