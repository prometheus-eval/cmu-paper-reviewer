#!/usr/bin/env python3
r"""
Build the LaTeX appendix listing every categorized S/W comment.

For each W1-W16 / S1-S6 code and the residual buckets, enumerate every
expert annotator comment in that bucket, grouped by source (item-level
then paper-level), with full text and traceable citation fields.

Output: analysis/main_results/appendix_sw_comments.tex

The produced `.tex` is a drop-in `\input{}` file — it starts with
`\section*{...}` and contains `longtable` blocks for each category.
The caller should include the `longtable` package.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
JSON_PATH = _HERE / 'reviewer_sw_analysis.json'
OUT = _HERE / 'appendix_sw_comments.tex'


MODEL_NAMES = {'GPT': 'GPT-5.2', 'Claude': 'Claude 4.5', 'Gemini': 'Gemini 3.0'}


# Non-Latin source comments with hand-verified English translations.
# Keyed by comment_id so we can substitute deterministically before TeX
# escaping. The Korean entries below are from a Korean-language paper-15
# reviewer on the secondary annotator pass. The English translation
# replaces the original-language text entirely, with an "[English
# translation from Korean]" note so the provenance is explicit.
TRANSLATIONS = {
    'P_p15_Claude_slot2_secondary':
        '[English translation from Korean] '
        'Argued the problem of the absence of a healthy control group in '
        'the most systematic way.',
    'P_p15_GPT_slot1_secondary':
        '[English translation from Korean] '
        'Code-level analysis is unique. These are problems that cannot be '
        'found by reading the paper text alone -- they are the result of '
        'analyzing the actual codebase.',
    'P_p15_Gemini_slot3_secondary':
        '[English translation from Korean] '
        'Clearly distinguishes sleep-structure abnormalities (local sleep '
        'disruption) from circadian-clock dysfunction.',
}


CATEGORY_ORDER = [
    ('W1',  'Missing community / field norms'),
    ('W2',  'Over-harsh / out-of-scope / unrealistic'),
    ('W3',  'Paper explicitly states X, AI says missing'),
    ('W4',  'Redundancy across the 3 AI reviewers'),
    ('W5',  'Vague / verbose / no actionable recommendation'),
    ('W6',  'Trivial / nitpicking'),
    ('W7',  'Technical term-of-art confusion'),
    ('W8',  'Cites evidence from after the preprint'),
    ('W9',  'Over-inflates small code/text inconsistencies'),
    ('W10', 'Criticizes what authors already flagged as a limitation'),
    ('W11', 'AI misreads a figure or caption'),
    ('W12', 'AI misquotes or fabricates a verbatim quote'),
    ('W13', 'AI misses supplementary content'),
    ('W14', "Ignores authors' own cited prior work"),
    ('W15', 'AI misreads a table'),
    ('W16', 'Cannot analyze figures --- only text'),
    ('W_unspecified', 'Residual: AI judged Not Correct without specific reason'),
    ('S1', 'Statistical / methodology rigor'),
    ('S2', 'Code reading'),
    ('S3', 'Specialized niche field catch'),
    ('S4', 'Internal consistency across sections'),
    ('S5', 'Reproducibility / dependency failures'),
    ('S6', 'Big-picture / counter-narrative synthesis'),
    ('S_generic', 'Residual: AI judged Correct without specific reason'),
]


def tex_escape(s: str) -> str:
    """Escape LaTeX special characters while preserving readability."""
    if not s:
        return ''
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Order matters: backslash first
    repl = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
        ('<', r'\textless{}'),
        ('>', r'\textgreater{}'),
        ('"', "''"),
        # Non-ASCII common chars
        ('—', '---'),
        ('–', '--'),
        ('‘', "`"),
        ('’', "'"),
        ('“', "``"),
        ('”', "''"),
        ('…', r'\dots{}'),
        ('°', r'$^{\circ}$'),
        ('≥', r'$\geq$'),
        ('≤', r'$\leq$'),
        ('×', r'$\times$'),
        ('²', '$^{2}$'),
        ('³', '$^{3}$'),
        ('α', r'$\alpha$'),
        ('β', r'$\beta$'),
        ('γ', r'$\gamma$'),
        ('δ', r'$\delta$'),
        ('μ', r'$\mu$'),
        ('π', r'$\pi$'),
        ('σ', r'$\sigma$'),
        ('ω', r'$\omega$'),
        ('Δ', r'$\Delta$'),
        ('Ω', r'$\Omega$'),
    ]
    for old, new in repl:
        s = s.replace(old, new)
    # Drop any remaining non-Latin characters that would break pdflatex.
    # Non-Latin (e.g. Korean, CJK) entries are handled up-front via the
    # TRANSLATIONS dict so the substantive English content survives —
    # anything still non-Latin at this point is a stray glyph, not signal.
    out_chars = []
    for ch in s:
        if ord(ch) < 128:
            out_chars.append(ch)
        elif ch in '\xa0':
            out_chars.append(' ')
        else:
            # Keep accented Latin characters that pdflatex can render,
            # drop CJK / other ideographs (the semantic content of any
            # non-Latin comment is already substituted via TRANSLATIONS).
            try:
                import unicodedata
                name = unicodedata.name(ch, '')
                if name.startswith('LATIN'):
                    out_chars.append(ch)
                else:
                    # Silently drop: the English translation is already
                    # in place, so any remaining glyphs are from the
                    # bracketed original-language quote
                    pass
            except ValueError:
                pass
    return ''.join(out_chars)


def format_citation(e):
    if e['source'] == 'item':
        return (f"P{e['paper_id']} $\\cdot$ "
                f"{MODEL_NAMES.get(e['reviewer_id'], e['reviewer_id'])} $\\cdot$ "
                f"item {e['item_number']} $\\cdot$ {e['annotator_source']}")
    else:
        return (f"P{e['paper_id']} $\\cdot$ "
                f"{MODEL_NAMES.get(e['reviewer_id'], e['reviewer_id'])} $\\cdot$ "
                f"paper-level slot {e['ai_slot']} $\\cdot$ {e['annotator_source']}")


def main():
    data = json.loads(JSON_PATH.read_text())
    entries = data['entries']

    by_code = defaultdict(list)
    for e in entries:
        by_code[e['final_label']].append(e)

    counts = {code: len(by_code[code]) for code, _ in CATEGORY_ORDER}
    total = sum(counts.values())

    lines = []
    lines.append('% Auto-generated by analysis/build_appendix_sw.py')
    lines.append('% Requires: \\usepackage{longtable}, \\usepackage{array}')
    lines.append('')
    lines.append('\\section*{Appendix: All Categorized Expert-Annotator Comments}')
    lines.append('')
    lines.append(
        'This appendix enumerates every expert annotator comment that '
        'participated in the S/W (strengths / weaknesses) classification '
        f'of AI reviews. The full corpus is {total} substantive comments '
        '(321 item-level AI comments from '
        '\\texttt{expert\\_comments\\_item\\_level.json} plus 121 '
        'paper-level descriptive comments from '
        '\\texttt{expert\\_comments\\_paper\\_level.json}, routed via '
        '\\texttt{reviewer\\_unique\\_items.json}). Comments labelled as '
        'being about a human reviewer, or carrying explicit '
        '\\emph{item-number} references instead of free-form prose, are '
        'handled by a separate artifact and are not listed here.'
    )
    lines.append('')
    lines.append(
        'Within each category, comments are sorted first by source '
        '(item-level before paper-level), then by paper id and reviewer. '
        'Each row is citable via its paper id, reviewer, and item number '
        'or paper-level slot number, so any comment can be traced back to '
        'the source JSON.'
    )
    lines.append('')
    lines.append(
        'Codes have been renumbered by final frequency. $W_{16}$ '
        '(``Cannot analyze figures --- only text\'\') previously carried '
        'the name $W_{17}$; the old $W_{16}$ (``Aggressive / ML-style '
        'tone in non-CS fields\'\') is empty after manual verification '
        'and has been removed from the taxonomy.'
    )
    lines.append('')

    # Summary table — need to escape underscores in residual code names
    def code_for_tex(c):
        return c.replace('_', r'\_')

    lines.append('\\subsection*{Category summary}')
    lines.append('\\begin{center}')
    lines.append('\\begin{tabular}{l l r}')
    lines.append('\\hline')
    lines.append('\\textbf{Code} & \\textbf{Name} & \\textbf{n} \\\\')
    lines.append('\\hline')
    for code, name in CATEGORY_ORDER:
        n = counts[code]
        lines.append(f'{code_for_tex(code)} & {name} & {n} \\\\')
    lines.append('\\hline')
    lines.append(f'\\textbf{{Total}} & & \\textbf{{{total}}} \\\\')
    lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append('\\end{center}')
    lines.append('')

    # One longtable per category
    for code, name in CATEGORY_ORDER:
        rows = by_code.get(code, [])
        if not rows:
            lines.append(f'\\subsection*{{{code_for_tex(code)}: {name} --- n = 0}}')
            lines.append('')
            lines.append('No comments in this category.')
            lines.append('')
            continue

        # Sort: item first, then paper; within each by paper_id / reviewer / item
        rows.sort(key=lambda r: (
            0 if r['source'] == 'item' else 1,
            r['paper_id'],
            r['reviewer_id'],
            r.get('item_number', r.get('ai_slot', 0)),
            r['annotator_source'],
        ))

        lines.append(f'\\subsection*{{{code_for_tex(code)}: {name} --- n = {len(rows)}}}')
        lines.append('')
        lines.append('{\\footnotesize')
        lines.append('\\begin{longtable}{@{}p{3.2cm}p{11.5cm}@{}}')
        lines.append('\\toprule')
        lines.append('\\textbf{Citation} & \\textbf{Expert comment} \\\\')
        lines.append('\\midrule')
        lines.append('\\endfirsthead')
        lines.append('\\toprule')
        lines.append('\\textbf{Citation} & \\textbf{Expert comment} \\\\')
        lines.append('\\midrule')
        lines.append('\\endhead')
        lines.append('\\bottomrule')
        lines.append('\\endlastfoot')

        for r in rows:
            cite = format_citation(r)
            # Apply English translation if this entry is non-Latin
            raw_text = TRANSLATIONS.get(r['comment_id'], r['text'])
            body = tex_escape(raw_text)
            if not body:
                body = r'\emph{(empty)}'
            lines.append(f'{cite} & {body} \\\\[3pt]')

        lines.append('\\end{longtable}')
        lines.append('}')
        lines.append('')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {OUT}')
    print(f'  total categorized comments: {total}')
    for code, name in CATEGORY_ORDER:
        print(f'  {code:<16} {counts[code]:>4}  {name}')


if __name__ == '__main__':
    main()
