#!/usr/bin/env python3
"""
Figure 5 — flagship case studies for the top-3 AI-reviewer weaknesses
and strengths, after frequency renumbering.

Layout: 2 rows × 3 columns of text-flow panels, tightly packed.
Top row    (weaknesses): W1 Field norms | W2 Over-harsh | W3 Paper-states-X
Bottom row (strengths):  S1 Stat rigor  | S2 Code reading | S3 Niche catch

Each panel shows a compact category header and 3 hand-picked expert
excerpts. Each excerpt has a [AI] line and an [Expert] line so the
reader can read the failure/strength pattern in one or two glances.
Every excerpt carries its paper_id + reviewer id + item/slot so it can
be traced back to the source JSON.

Output: analysis/main_results/figure5.svg (+ .png)

Data source: reviewer_sw_analysis.json (post-verification, frequency-
renumbered).
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
OUT_DIR = _HERE


MODEL_NAMES = {'GPT': 'GPT-5.2', 'Claude': 'Claude 4.5', 'Gemini': 'Gemini 3.0'}


# -----------------------------------------------------------------------
# FLAGSHIP EXCERPTS
#
# Each entry is (paper_id, reviewer, item_or_slot, source, ai_line, expert_line).
# Both lines are hand-trimmed one-sentence summaries. The layout assumes
# each example fits in 2 + 2 wrapped lines at width 62.
# -----------------------------------------------------------------------

WEAKNESS_BLOCKS = [
    ('W1', 'Missing field norms / community context', 54, [
        (3, 'Claude', 4, 'primary',
         '1D corrections are unreliable; authors should use higher-level '
         'methods.',
         'It is well known that 1D corrections are not working well and '
         'that is the reason researchers use better approximations like '
         'RPMD and Instanton theory.'),
        (8, 'GPT', 1, 'primary',
         'SPlot is an unusual choice and needs justification.',
         'SPlot is a common technique in particle physics and widely '
         'used in many publications.'),
        (15, 'Claude', 2, 'primary',
         'No healthy controls -- requires a fix.',
         "It's not possible to obtain intracranial data from healthy "
         "controls -- it won't pass the ethical committee protocol."),
    ]),

    ('W2', 'Over-harsh / out-of-scope / unrealistic', 46, [
        (6, 'Claude', 4, 'primary',
         'Follow-up duration is too short compared to Grant et al.',
         'The safety aspects belong in another paper; 1 week vs 2-4 '
         'weeks makes no difference for the MR engineering.'),
        (29, 'Claude', 1, 'primary',
         'The proof would benefit from a broader validation study.',
         'This is technically correct, but outside the scope of the '
         'paper.'),
        (46, 'Claude', 2, 'primary',
         'Accuracy on out-of-range values is a concern.',
         "Yes, but what can you do? -- you can't evaluate on values "
         "that don't exist in the training set."),
    ]),

    ('W3', 'Paper explicitly states X, AI claims missing', 37, [
        (1, 'Gemini', 1, 'primary',
         'Paper should employ finite-size correction.',
         'The authors write explicitly that they employ finite-size '
         'correction.'),
        (4, 'GPT', 3, 'primary',
         'The code is not available in the linked repository.',
         'The authors provide the code; GitHub vs GitLab does not '
         'matter.'),
        (66, 'Claude', 2, 'primary',
         'Measurements should be repeated; averaged results needed.',
         'The authors mention repetition of measurements and using '
         'averaged results in the end.'),
    ]),
]


STRENGTH_BLOCKS = [
    ('S1', 'Statistical / methodology rigor', 45, [
        (35, 'GPT', 1, 'primary',
         'The PPI trainer has no validation split; model selection uses '
         'test metrics, misleadingly named best_valid_f1.',
         'A data-leakage catch that none of the human reviewers '
         'flagged.'),
        (60, 'Claude', 3, 'secondary',
         'Treating correlated regional observations within the same '
         'event as independent violates the K-S test assumptions.',
         'A genuinely useful statistical critique that neither the first '
         'reviewer nor the human reviewers identified.'),
        (47, 'Claude', 3, 'paper',
         'The entropy equation in the crypto-key section is '
         'mathematically incorrect.',
         'AI #3 uniquely pointed out that the entropy equation is '
         'mathematically incorrect.'),
    ]),

    ('S2', 'Code reading -- AI opens the repo, not just the PDF', 28, [
        (49, 'GPT', 2, 'secondary',
         'Opened the GitHub repo and found the claim of 1,000 random '
         'iterations is actually 68 deterministic splits.',
         'This reviewer actually visited the GitHub repository, '
         'dissected and analyzed the Python scripts.'),
        (35, 'Gemini', 1, 'primary',
         'Data leakage is real; trainer_dpi.py uses test metrics for '
         'model selection despite having a validation set.',
         'Code verification fully confirms the data leakage claim.'),
        (1, 'GPT', 4, 'primary',
         '(Repository inspection: flags parts of the paper not '
         'properly discussed.)',
         'I find it quite impressive that this reviewer actually looks '
         'at the code -- I am not doing it on a regular basis.'),
    ]),

    ('S3', 'Specialized niche field catch', 27, [
        (64, 'GPT', 4, 'primary',
         'Retrieving a complex field (with the imaginary part) through '
         'multimode fiber propagation is ill-posed.',
         'This is a good catch: to the best of my knowledge there is '
         'no method to do that well with multimode fiber propagation.'),
        (77, 'GPT', 2, 'primary',
         'The substrate for compound 4v contains a stereogenic center '
         '-- diastereoselectivity is critical.',
         'Presumably mixture of diastereomers were formed in compound '
         '4v; it was not mentioned by the human reviewers.'),
        (52, 'Gemini', 2, 'paper',
         'Biosafety of gold dissolution has not been addressed.',
         'Dissolution of gold for drug release can be toxic -- needs '
         'testing.'),
    ]),
]


# -----------------------------------------------------------------------
# RENDER
# -----------------------------------------------------------------------

AI_COLOR = '#666666'
EXPERT_COLOR_W = '#8F2929'
EXPERT_COLOR_S = '#23487A'
CITE_COLOR = '#555555'
HEADER_COLOR_W = '#C24C4C'
HEADER_COLOR_S = '#3B6EA6'
DIVIDER_COLOR = '#DADADA'

WRAP_WIDTH = 58   # characters per wrapped line
AI_LABEL = 'AI:'
EXP_LABEL = 'Expert:'
FS_HEADER_CODE = 13
FS_HEADER_NAME = 10.5
FS_CITE = 9
FS_BODY = 9
LINE_H = 0.048    # fractional panel height per wrapped text line


def render_panel(ax, code, name, n, examples, kind):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    header_color = HEADER_COLOR_W if kind == 'W' else HEADER_COLOR_S
    exp_color = EXPERT_COLOR_W if kind == 'W' else EXPERT_COLOR_S

    # Top header line — use mathtext for the bold code so matplotlib
    # handles the spacing automatically.
    y_header = 0.965
    header_text = rf'$\mathbf{{{code}}}$   {name}'
    ax.text(0.015, y_header, header_text, fontsize=FS_HEADER_NAME,
            fontweight='bold', color='#222', va='top', ha='left',
            transform=ax.transAxes)
    # Draw the code again in red on top so it retains its color —
    # the mathtext version is black.
    ax.text(0.015, y_header, code, fontsize=FS_HEADER_NAME,
            fontweight='bold', color=header_color, va='top', ha='left',
            transform=ax.transAxes)
    ax.text(0.985, y_header, f'n = {n}', fontsize=FS_HEADER_NAME - 1,
            color='#555', va='top', ha='right', transform=ax.transAxes)

    # Header rule
    ax.plot([0.01, 0.99], [0.935, 0.935], color=header_color,
            lw=1.1, transform=ax.transAxes, clip_on=False)

    # Stack examples from the top down, measuring wrap-height on the fly
    y = 0.905
    for i, (pid, rev, itm, ann, ai_text, exp_text) in enumerate(examples):
        if i > 0:
            # divider line
            ax.plot([0.03, 0.97], [y + 0.012, y + 0.012],
                    color=DIVIDER_COLOR, lw=0.5,
                    transform=ax.transAxes, clip_on=False)
            y -= 0.014

        # Citation line
        if ann == 'paper':
            cite = (f'Paper {pid} · {MODEL_NAMES.get(rev, rev)} · '
                    f'paper-level')
        else:
            cite = (f'Paper {pid} · {MODEL_NAMES.get(rev, rev)} · '
                    f'item {itm} · {ann}')
        ax.text(0.015, y, cite, fontsize=FS_CITE, fontweight='bold',
                color=CITE_COLOR, va='top', ha='left', transform=ax.transAxes)
        y -= LINE_H * 1.0

        # AI line (wrapped)
        ai_lines = textwrap.wrap(ai_text, width=WRAP_WIDTH)
        ax.text(0.015, y, AI_LABEL, fontsize=FS_BODY, fontweight='bold',
                color=AI_COLOR, va='top', ha='left', transform=ax.transAxes)
        ax.text(0.08, y, '\n'.join(ai_lines), fontsize=FS_BODY,
                color='#222', va='top', ha='left', linespacing=1.18,
                transform=ax.transAxes)
        y -= LINE_H * len(ai_lines) + 0.006

        # Expert line (wrapped, hanging indent at same position)
        exp_lines = textwrap.wrap(exp_text, width=WRAP_WIDTH)
        ax.text(0.015, y, EXP_LABEL, fontsize=FS_BODY, fontweight='bold',
                color=exp_color, va='top', ha='left', transform=ax.transAxes)
        ax.text(0.08, y, '\n'.join(exp_lines), fontsize=FS_BODY,
                color='#222', va='top', ha='left', linespacing=1.18,
                transform=ax.transAxes)
        y -= LINE_H * len(exp_lines) + 0.012

    # Outer border
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='#CCC', lw=0.6,
            transform=ax.transAxes, clip_on=False)


def main():
    fig, axes = plt.subplots(
        2, 3,
        figsize=(16.2, 10.4),
        gridspec_kw={'wspace': 0.06, 'hspace': 0.10},
    )

    for ax, (code, name, n, examples) in zip(axes[0], WEAKNESS_BLOCKS):
        render_panel(ax, code, name, n, examples, 'W')

    for ax, (code, name, n, examples) in zip(axes[1], STRENGTH_BLOCKS):
        render_panel(ax, code, name, n, examples, 'S')

    fig.suptitle(
        'Flagship examples: top-3 AI-reviewer weaknesses (top row) and '
        'strengths (bottom row)',
        fontsize=13, fontweight='bold', y=0.985,
    )

    # Row labels on the far left
    fig.text(0.003, 0.73, 'Top-3\nweaknesses', fontsize=11,
             fontweight='bold', color=HEADER_COLOR_W, rotation=0,
             va='center', ha='left')
    fig.text(0.003, 0.27, 'Top-3\nstrengths', fontsize=11,
             fontweight='bold', color=HEADER_COLOR_S, rotation=0,
             va='center', ha='left')

    fig.tight_layout(rect=(0.025, 0, 1, 0.965))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / 'figure5.svg'
    png_path = OUT_DIR / 'figure5.png'
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {svg_path}')
    print(f'wrote {png_path}')


if __name__ == '__main__':
    main()
