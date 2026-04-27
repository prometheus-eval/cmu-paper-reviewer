#!/usr/bin/env python3
"""
Figure 4 — bar chart of AI-reviewer strength / weakness categories.

For each specific W1-W16 and S1-S6 code, show total count as a
horizontal bar, stacked by source (item-level vs paper-level) to make
the survey-design bias visible. Bars are ordered by total count
descending within each panel. Residual buckets (W_unspecified,
S_generic) are NOT shown — the figure is about the specific,
interpretable categories only.

Output: analysis/main_results/figure4.svg (+ .png)

Data source: reviewer_sw_analysis.json (post-verification,
frequency-renumbered). 260 W + 132 S specific tags.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = Path(__file__).resolve().parent
# reviewer_sw_analysis.json is co-located in main_results/ as the figure's
# data input. It is a pre-built qualitative-analysis artifact — practitioners
# do not re-build it; the construction pipeline lives in the private
# llm-as-a-reviewer/peerreviewbench/ repo.
JSON_PATH = _HERE / 'reviewer_sw_analysis.json'
OUT_DIR = _HERE


# Short labels for the bars (<~55 chars to render comfortably)
W_LABELS = {
    'W1':  'Missing field norms / community context',
    'W2':  'Over-harsh / out-of-scope / unrealistic',
    'W3':  'Paper explicitly states X, AI says missing',
    'W4':  'Redundancy across the 3 AI reviewers',
    'W5':  'Vague / verbose / no actionable rec.',
    'W6':  'Trivial / nitpicking (typo, style)',
    'W7':  'Technical term-of-art confusion',
    'W8':  'Cites evidence that appeared after the preprint',
    'W9':  'Over-inflates small code/text mismatches',
    'W10': 'Criticizes what authors already flagged',
    'W11': 'AI misreads a figure or caption',
    'W12': 'AI misquotes / fabricates a verbatim quote',
    'W13': 'AI misses supplementary content',
    'W14': "Ignores authors' own cited prior work",
    'W15': 'AI misreads a table',
    'W16': 'Cannot analyze figures — only text',
}

S_LABELS = {
    'S1': 'Statistical / methodology rigor',
    'S2': 'Code reading (opens the repo, not just PDF)',
    'S3': 'Specialized niche field catch',
    'S4': 'Internal consistency across sections',
    'S5': 'Reproducibility / dependency failures',
    'S6': 'Big-picture / counter-narrative synthesis',
}


# Colors — matched to §2.4 category box styles
COL_ITEM_W = '#B4463C'   # muted terracotta
COL_PAPER_W = '#DC9B91'  # lighter tint
COL_ITEM_S = '#3C784B'   # muted forest green
COL_PAPER_S = '#91C3A0'  # lighter tint


def load_counts():
    data = json.loads(JSON_PATH.read_text())
    entries = data['entries']

    item_cnt = Counter(e['final_label'] for e in entries if e['source'] == 'item')
    paper_cnt = Counter(e['final_label'] for e in entries if e['source'] == 'paper')
    return entries, item_cnt, paper_cnt


def main():
    entries, item_cnt, paper_cnt = load_counts()
    total = len(entries)

    # Ordered W codes (descending total) — specific only
    w_codes = sorted(
        W_LABELS.keys(),
        key=lambda k: -(item_cnt.get(k, 0) + paper_cnt.get(k, 0)),
    )
    s_codes = sorted(
        S_LABELS.keys(),
        key=lambda k: -(item_cnt.get(k, 0) + paper_cnt.get(k, 0)),
    )

    w_n = sum(item_cnt.get(c, 0) + paper_cnt.get(c, 0) for c in w_codes)
    s_n = sum(item_cnt.get(c, 0) + paper_cnt.get(c, 0) for c in s_codes)

    fig, (ax_w, ax_s) = plt.subplots(
        1, 2,
        figsize=(22, 12),
        gridspec_kw={'width_ratios': [1.05, 0.95], 'wspace': 0.55},
    )

    # ---- WEAKNESSES ----------------------------------------------------
    w_item = [item_cnt.get(c, 0) for c in w_codes]
    w_paper = [paper_cnt.get(c, 0) for c in w_codes]
    w_total = [i + p for i, p in zip(w_item, w_paper)]
    y_w = list(range(len(w_codes)))

    ax_w.barh(y_w, w_item, color=COL_ITEM_W, label='_nolegend_',
              edgecolor='white', linewidth=0.8)
    ax_w.barh(y_w, w_paper, left=w_item, color=COL_PAPER_W, label='_nolegend_',
              edgecolor='white', linewidth=0.8)

    for yi, ti, ii, pi_ in zip(y_w, w_total, w_item, w_paper):
        if ti == 0:
            continue
        ax_w.text(ti + 0.6, yi, f'{ti}', va='center', fontsize=14,
                  fontweight='bold', color='#222')
        if ii > 0 and pi_ > 0:
            split_txt = f'  ({ii}/{pi_})'
            ax_w.text(ti + 3.1, yi, split_txt, va='center', fontsize=12, color='#333')

    ax_w.set_yticks(y_w)
    # Bold category code + normal-weight name using mathtext
    y_labels_w = [
        rf'$\mathbf{{{c}}}$  {W_LABELS[c]}' for c in w_codes
    ]
    ax_w.set_yticklabels(y_labels_w, fontsize=14)
    ax_w.invert_yaxis()
    ax_w.set_xlabel('Number of expert annotator comments', fontsize=15)
    ax_w.set_title(
        f'Weaknesses of AI reviewers (n = {w_n})',
        fontsize=16, fontweight='bold', pad=14,
    )
    ax_w.set_xlim(0, max(w_total) * 1.22)
    ax_w.spines['top'].set_visible(False)
    ax_w.spines['right'].set_visible(False)
    ax_w.tick_params(axis='x', labelsize=13)
    ax_w.grid(axis='x', alpha=0.25, linestyle=':')

    # ---- STRENGTHS -----------------------------------------------------
    s_item = [item_cnt.get(c, 0) for c in s_codes]
    s_paper = [paper_cnt.get(c, 0) for c in s_codes]
    s_total = [i + p for i, p in zip(s_item, s_paper)]
    y_s = list(range(len(s_codes)))

    ax_s.barh(y_s, s_item, color=COL_ITEM_S, label='_nolegend_',
              edgecolor='white', linewidth=0.8)
    ax_s.barh(y_s, s_paper, left=s_item, color=COL_PAPER_S, label='_nolegend_',
              edgecolor='white', linewidth=0.8)

    for yi, ti, ii, pi_ in zip(y_s, s_total, s_item, s_paper):
        if ti == 0:
            continue
        ax_s.text(ti + 0.6, yi, f'{ti}', va='center', fontsize=14,
                  fontweight='bold', color='#222')
        if ii > 0 and pi_ > 0:
            split_txt = f'  ({ii}/{pi_})'
            ax_s.text(ti + 3.1, yi, split_txt, va='center', fontsize=12, color='#333')

    ax_s.set_yticks(y_s)
    y_labels_s = [
        rf'$\mathbf{{{c}}}$  {S_LABELS[c]}' for c in s_codes
    ]
    ax_s.set_yticklabels(y_labels_s, fontsize=14)
    ax_s.invert_yaxis()
    ax_s.set_xlabel('Number of expert annotator comments', fontsize=15)
    ax_s.set_title(
        f'Strengths of AI reviewers (n = {s_n})',
        fontsize=16, fontweight='bold', pad=14,
    )
    ax_s.set_xlim(0, max(s_total) * 1.22)
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)
    ax_s.tick_params(axis='x', labelsize=13)
    ax_s.grid(axis='x', alpha=0.25, linestyle=':')

    # Legend — centered on the full figure, below both panels
    legend_handles = [
        mpatches.Patch(facecolor=COL_ITEM_W, edgecolor='white',
                       label='Weakness — item-level'),
        mpatches.Patch(facecolor=COL_PAPER_W, edgecolor='white',
                       label='Weakness — paper-level'),
        mpatches.Patch(facecolor=COL_ITEM_S, edgecolor='white',
                       label='Strength — item-level'),
        mpatches.Patch(facecolor=COL_PAPER_S, edgecolor='white',
                       label='Strength — paper-level'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=4,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f'Expert-annotator classification of AI-reviewer behaviors '
        f'(n = {w_n + s_n} specific S/W tags across 82 papers, 3 AI reviewers)',
        fontsize=18, fontweight='bold', y=0.995,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / 'figure4.svg'
    png_path = OUT_DIR / 'figure4.png'
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {svg_path}')
    print(f'wrote {png_path}')
    print(f'  n (weaknesses) = {w_n}')
    print(f'  n (strengths)  = {s_n}')
    print(f'  n (total)      = {w_n + s_n}')


if __name__ == '__main__':
    main()
