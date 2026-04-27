#!/usr/bin/env python3
"""
Generate table9.tex -- summary table combining key coverage numbers:
  - Human items covered by 1 / 3 AI reviewers (all and fully-good)
  - AI items covered by >=1 human reviewer
  - Uncovered AI item statistics (fraction, quality)

Paper-level bootstrap (10,000 iterations) for all 95% CIs.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "outputs", "full_similarity", "pairs_llm_azure_ai__gpt-5_4.jsonl",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_TEX = os.path.join(OUT_DIR, "table9.tex")

N_BOOT = 10_000
SEED = 42

AI_REVIEWERS = ["Claude", "GPT", "Gemini"]

ORDINAL = {
    "different subject": 0,
    "same subject, different argument": 1,
    "same subject, same argument, different evidence": 2,
    "same subject, same argument, same evidence": 3,
}


# Papers dropped due to license restrictions (not confirmed CC BY 4.0)
DROPPED_PAPER_IDS = {11, 20, 22}

from rogan_gladen import rogan_gladen_correct, resample_sens_spec


# ── load similarity pairs ────────────────────────────────────────────
def load_pairs(path):
    pairs = []
    n_err = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["error"] is not None:
                n_err += 1
                continue
            if r.get("paper_id") in DROPPED_PAPER_IDS:
                continue
            pairs.append(r)
    print(f"Loaded {len(pairs)} valid pairs, skipped {n_err} errors.")
    return pairs


# ── load expert annotations (primary only) ───────────────────────────
def load_annotations():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    from load_data import load_expert_annotation_rows
    rows = load_expert_annotation_rows()
    primary = [r for r in rows if r["annotator_source"] == "primary"]
    return primary


def is_fully_good(row):
    return (row.get("correctness") == "Correct"
            and row.get("significance") == "Significant"
            and row.get("evidence") == "Sufficient")


# ── build all data structures ────────────────────────────────────────
def build_data(pairs, annotations):
    """Build all data structures needed for the table metrics."""
    # Annotation lookup
    annot_lookup = {}
    for r in annotations:
        key = (r["paper_id"], r["reviewer_id"], r["review_item_number"])
        annot_lookup[key] = r

    # Fully-good set
    fully_good_set = set()
    for r in annotations:
        if is_fully_good(r):
            fully_good_set.add((r["paper_id"], r["reviewer_id"], r["review_item_number"]))

    # ── Coverage structures (from H-A pairs) ──
    human_items_by_paper = defaultdict(set)
    ai_items_by_paper = defaultdict(set)
    h_covered_by_ai = defaultdict(lambda: defaultdict(set))
    ai_max_sim = defaultdict(int)

    for p in pairs:
        if p["pair_type"] != "H-A":
            continue
        pid = p["paper_id"]
        h_rev = p["item_a"]["reviewer_id"]
        h_item = p["item_a"]["review_item_number"]
        ai_rev = p["item_b"]["reviewer_id"]
        ai_item = p["item_b"]["review_item_number"]

        human_items_by_paper[pid].add((h_rev, h_item))
        ai_items_by_paper[pid].add((ai_rev, ai_item))

        if p["parsed_binary"] == "similar":
            h_covered_by_ai[pid][(h_rev, h_item)].add(ai_rev)

        ai_key = (pid, ai_rev, ai_item)
        ordinal = ORDINAL.get(p["parsed_answer"], 0)
        ai_max_sim[ai_key] = max(ai_max_sim.get(ai_key, 0), ordinal)

    human_items_by_paper = {pid: sorted(items) for pid, items in human_items_by_paper.items()}
    ai_items_by_paper = {pid: sorted(items) for pid, items in ai_items_by_paper.items()}

    # Classify AI items as matched/uncovered
    matched_ai_by_paper = defaultdict(list)
    uncovered_ai_by_paper = defaultdict(list)
    for pid, items in ai_items_by_paper.items():
        for (ai_rev, ai_item) in items:
            key = (pid, ai_rev, ai_item)
            if ai_max_sim.get(key, 0) >= 2:
                matched_ai_by_paper[pid].append(key)
            else:
                uncovered_ai_by_paper[pid].append(key)

    return {
        "human_items_by_paper": human_items_by_paper,
        "ai_items_by_paper": ai_items_by_paper,
        "fully_good_set": fully_good_set,
        "h_covered_by_ai": h_covered_by_ai,
        "matched_ai_by_paper": dict(matched_ai_by_paper),
        "uncovered_ai_by_paper": dict(uncovered_ai_by_paper),
        "annot_lookup": annot_lookup,
    }


# ── compute all table metrics for a set of paper_ids ────────────────
def compute_metrics(paper_ids, data):
    """Paper-level averaging: compute one rate per paper, then mean across papers."""
    human_items_by_paper = data["human_items_by_paper"]
    ai_items_by_paper = data["ai_items_by_paper"]
    fully_good_set = data["fully_good_set"]
    h_covered_by_ai = data["h_covered_by_ai"]
    matched_ai_by_paper = data["matched_ai_by_paper"]
    uncovered_ai_by_paper = data["uncovered_ai_by_paper"]
    annot_lookup = data["annot_lookup"]

    # Per-paper rates
    paper_1ai_all = []
    paper_3ai_all = []
    paper_1ai_fg = []
    paper_3ai_fg = []
    paper_ai_by_h = []
    paper_uncovered_frac = []
    paper_uncovered_fg = []
    paper_uncovered_correct = []

    for pid in paper_ids:
        h_items = human_items_by_paper.get(pid, [])
        a_items = ai_items_by_paper.get(pid, [])
        coverage_map = h_covered_by_ai.get(pid, {})

        if not h_items:
            continue

        all_ai = set(AI_REVIEWERS)

        # Group human items by reviewer
        by_h_rev = defaultdict(list)
        for (h_rev, h_item) in h_items:
            by_h_rev[h_rev].append((h_rev, h_item))

        # ── 1 AI coverage: per (h_rev, ai_rev) pair, then average ──
        pair_coverages = []
        pair_fg_coverages = []
        for h_rev, h_rev_items in by_h_rev.items():
            fg_items = [k for k in h_rev_items if (pid, k[0], k[1]) in fully_good_set]
            for ai_r in AI_REVIEWERS:
                cov = sum(1 for k in h_rev_items if ai_r in coverage_map.get(k, set()))
                pair_coverages.append(cov / len(h_rev_items))
                if fg_items:
                    fg_cov = sum(1 for k in fg_items if ai_r in coverage_map.get(k, set()))
                    pair_fg_coverages.append(fg_cov / len(fg_items))

        paper_1ai_all.append(np.mean(pair_coverages))
        if pair_fg_coverages:
            paper_1ai_fg.append(np.mean(pair_fg_coverages))

        # ── 3 AI coverage: per h_rev, coverage by union of 3 AIs, then average ──
        rev_3ai_coverages = []
        rev_3ai_fg_coverages = []
        for h_rev, h_rev_items in by_h_rev.items():
            fg_items = [k for k in h_rev_items if (pid, k[0], k[1]) in fully_good_set]
            cov_3 = sum(1 for k in h_rev_items if coverage_map.get(k, set()) & all_ai)
            rev_3ai_coverages.append(cov_3 / len(h_rev_items))
            if fg_items:
                fg_cov_3 = sum(1 for k in fg_items if coverage_map.get(k, set()) & all_ai)
                rev_3ai_fg_coverages.append(fg_cov_3 / len(fg_items))

        paper_3ai_all.append(np.mean(rev_3ai_coverages))
        if rev_3ai_fg_coverages:
            paper_3ai_fg.append(np.mean(rev_3ai_fg_coverages))

        # ── AI covered by human (paper-level) ──
        if a_items:
            n_matched = len(matched_ai_by_paper.get(pid, []))
            paper_ai_by_h.append(n_matched / len(a_items))

        # ── Uncovered stats (paper-level) ──
        uncovereds = uncovered_ai_by_paper.get(pid, [])
        if a_items:
            paper_uncovered_frac.append(len(uncovereds) / len(a_items))

        uncovered_fg_vals = []
        uncovered_corr_vals = []
        for key in uncovereds:
            row = annot_lookup.get(key)
            if row is None:
                continue
            uncovered_fg_vals.append(1 if is_fully_good(row) else 0)
            uncovered_corr_vals.append(1 if row["correctness"] == "Correct" else 0)
        if uncovered_fg_vals:
            paper_uncovered_fg.append(np.mean(uncovered_fg_vals))
        if uncovered_corr_vals:
            paper_uncovered_correct.append(np.mean(uncovered_corr_vals))

    results = {
        "1ai_all": np.mean(paper_1ai_all) if paper_1ai_all else 0.0,
        "3ai_all": np.mean(paper_3ai_all) if paper_3ai_all else 0.0,
        "1ai_fg": np.mean(paper_1ai_fg) if paper_1ai_fg else 0.0,
        "3ai_fg": np.mean(paper_3ai_fg) if paper_3ai_fg else 0.0,
        "ai_by_human": np.mean(paper_ai_by_h) if paper_ai_by_h else 0.0,
        "uncovered_frac": np.mean(paper_uncovered_frac) if paper_uncovered_frac else 0.0,
        "uncovered_fg": np.mean(paper_uncovered_fg) if paper_uncovered_fg else 0.0,
        "uncovered_correct": np.mean(paper_uncovered_correct) if paper_uncovered_correct else 0.0,
    }
    return results


# Coverage metrics that should receive Rogan-Gladen correction
# (these are binary-similarity-derived rates; uncovered quality metrics are not)
# Note: uncovered_frac is the complement of ai_by_human, so we derive it
# as 1 - corrected(ai_by_human) rather than correcting independently.
_RG_CORRECTABLE = {"1ai_all", "3ai_all", "1ai_fg", "3ai_fg",
                   "ai_by_human"}
_COMPLEMENT_OF = {"uncovered_frac": "ai_by_human"}


# ── paper-level bootstrap ───────────────────────────────────────────
def bootstrap_all(data, n_boot=N_BOOT, seed=SEED):
    all_pids = sorted(data["human_items_by_paper"].keys())
    n_papers = len(all_pids)
    paper_arr = np.array(all_pids)
    rng = np.random.RandomState(seed)

    raw_obs = compute_metrics(all_pids, data)
    metric_names = list(raw_obs.keys())

    # RG-corrected point estimates
    obs = {}
    for m in metric_names:
        if m in _RG_CORRECTABLE:
            obs[m] = rogan_gladen_correct(raw_obs[m])
        elif m in _COMPLEMENT_OF:
            obs[m] = 1.0 - rogan_gladen_correct(raw_obs[_COMPLEMENT_OF[m]])
        else:
            obs[m] = raw_obs[m]

    boot = {m: np.empty(n_boot) for m in metric_names}

    for b in range(n_boot):
        sampled = rng.choice(paper_arr, size=n_papers, replace=True)
        stats = compute_metrics(sampled, data)
        b_sens, b_spec = resample_sens_spec(rng)
        for m in metric_names:
            if m in _RG_CORRECTABLE:
                boot[m][b] = rogan_gladen_correct(stats[m], b_sens, b_spec)
            elif m in _COMPLEMENT_OF:
                boot[m][b] = 1.0 - rogan_gladen_correct(
                    stats[_COMPLEMENT_OF[m]], b_sens, b_spec)
            else:
                boot[m][b] = stats[m]

    cis = {}
    for m in metric_names:
        cis[m] = (np.percentile(boot[m], 2.5), np.percentile(boot[m], 97.5))

    return obs, cis


# ── LaTeX generation ─────────────────────────────────────────────────
def fmt_pct(val):
    return f"{val * 100:.1f}\\%"


def fmt_ci(ci):
    return f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"


def generate_table(obs, cis):
    rows = [
        ("Human items covered by 1 AI reviewer", "1ai_all"),
        ("Human items covered by 3 AI reviewers", "3ai_all"),
        ("Fully-good human items covered by 1 AI", "1ai_fg"),
        ("Fully-good human items covered by 3 AI", "3ai_fg"),
        (r"AI items covered by $\geq$1 human reviewer", "ai_by_human"),
        ("Uncovered AI items (no human match)", "uncovered_frac"),
        ("Uncovered AI items that are fully good", "uncovered_fg"),
        ("Uncovered AI items that are correct", "uncovered_correct"),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\fontsize{8}{10}\selectfont")
    lines.append(r"\begin{tabular}{l c c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Value} & \textbf{95\% CI} \\")
    lines.append(r"\midrule")

    for label, key in rows:
        val_str = fmt_pct(obs[key])
        ci_str = fmt_ci(cis[key])
        lines.append(f"{label} & {val_str} & {ci_str} \\\\")
        # Add midrule after the fully-good-3AI and after ai_by_human rows
        if key == "3ai_fg" or key == "ai_by_human":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Coverage summary (per-reviewer-pair averaging). "
        r"``Covered'' means at least one H--A pair is classified as similar "
        r"(near-paraphrase or convergent conclusion, ordinal $\geq$ 2). "
        r"``1 AI reviewer'' reports the mean coverage across all "
        r"(human reviewer, AI reviewer) pairs within each paper, then averaged across papers. "
        r"``3 AI reviewers'' reports per-human-reviewer coverage by the union of 3 AI reviewers, "
        r"averaged across human reviewers within each paper, then across papers. "
        r"``Uncovered'' AI items have no similar human counterpart. "
        r"All values are Rogan-Gladen-corrected; "
        r"95\% CIs from 10{,}000-iteration paper-level bootstrap with resampled sensitivity/specificity.}"
    )
    lines.append(r"\label{tab:coverage-summary}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


# ── main ─────────────────────────────────────────────────────────────
def main():
    pairs = load_pairs(DATA_FILE)
    annotations = load_annotations()
    data = build_data(pairs, annotations)

    total_h = sum(len(v) for v in data["human_items_by_paper"].values())
    total_a = sum(len(v) for v in data["ai_items_by_paper"].values())
    total_matched = sum(len(v) for v in data["matched_ai_by_paper"].values())
    total_uncovered = sum(len(v) for v in data["uncovered_ai_by_paper"].values())
    print(f"\nHuman items: {total_h}")
    print(f"AI items: {total_a}")
    print(f"Matched AI: {total_matched}")
    print(f"Uncovered AI: {total_uncovered}")

    obs, cis = bootstrap_all(data)

    print("\n=== Table 9: Coverage Summary ===")
    row_labels = {
        "1ai_all": "Human items covered by 1 AI reviewer",
        "3ai_all": "Human items covered by 3 AI reviewers",
        "1ai_fg": "Fully-good human items covered by 1 AI",
        "3ai_fg": "Fully-good human items covered by 3 AI",
        "ai_by_human": "AI items covered by >=1 human reviewer",
        "uncovered_frac": "Uncovered AI items (no human match)",
        "uncovered_fg": "Uncovered AI items that are fully good",
        "uncovered_correct": "Uncovered AI items that are correct",
    }
    for key, label in row_labels.items():
        ci = cis[key]
        print(f"  {label}: {obs[key]*100:.1f}%  95% CI [{ci[0]*100:.1f}, {ci[1]*100:.1f}]")

    tex = generate_table(obs, cis)
    with open(OUT_TEX, "w") as f:
        f.write(tex)
    print(f"\nWrote {OUT_TEX}")


if __name__ == "__main__":
    main()
