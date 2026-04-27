#!/usr/bin/env python3
"""
Generate table8.tex -- 3-way similarity headline table with same/different
reviewer H-H and A-A splits, including 4-way category breakdown.

Uses paper-level bootstrap (10,000 iterations) for 95% CIs.
"""

import json
import os
import numpy as np
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "outputs", "full_similarity", "pairs_llm_azure_ai__gpt-5_4.jsonl",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_TEX = os.path.join(OUT_DIR, "table8.tex")

N_BOOT = 10_000
SEED = 42

# 4-way answer labels
NEAR_PARAPHRASE = "same subject, same argument, same evidence"
CONVERGENT = "same subject, same argument, different evidence"
TOPICAL = "same subject, different argument"
UNRELATED = "different subject"

CATEGORY_ORDER = [NEAR_PARAPHRASE, CONVERGENT, TOPICAL, UNRELATED]
CATEGORY_SHORT = {
    NEAR_PARAPHRASE: "near-para",
    CONVERGENT: "convergent",
    TOPICAL: "topical",
    UNRELATED: "unrelated",
}

# Papers dropped due to license restrictions (not confirmed CC BY 4.0)
DROPPED_PAPER_IDS = {11, 20, 22}

from rogan_gladen import rogan_gladen_correct, resample_sens_spec


# ── load data ──────────────────────────────────────────────────────────
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


# ── grouping ───────────────────────────────────────────────────────────
def group_pairs(pairs):
    """Return dict: group_name -> list of pairs."""
    groups = {
        "H-H (all)": [],
        "H-H (same reviewer)": [],
        "H-H (diff reviewer)": [],
        "A-A (all)": [],
        "A-A (same model)": [],
        "A-A (diff model)": [],
        "H-A": [],
    }
    for p in pairs:
        pt = p["pair_type"]
        sr = p["same_reviewer"]
        if pt == "H-H":
            groups["H-H (all)"].append(p)
            if sr:
                groups["H-H (same reviewer)"].append(p)
            else:
                groups["H-H (diff reviewer)"].append(p)
        elif pt == "A-A":
            groups["A-A (all)"].append(p)
            if sr:
                groups["A-A (same model)"].append(p)
            else:
                groups["A-A (diff model)"].append(p)
        elif pt == "H-A":
            groups["H-A"].append(p)
    return groups


# ── paper-level bootstrap ─────────────────────────────────────────────
def paper_bootstrap(pairs, n_boot=N_BOOT, seed=SEED):
    """Paper-level bootstrap for P(similar) and 4-way fractions.

    Returns dict with keys: p_sim, ci_sim, p_near, ci_near, p_conv, ci_conv,
    p_top, ci_top, p_unr, ci_unr.
    """
    # Index pairs by paper
    by_paper = defaultdict(list)
    for p in pairs:
        by_paper[p["paper_id"]].append(p)

    paper_ids = sorted(by_paper.keys())
    n_papers = len(paper_ids)

    if n_papers == 0:
        return {
            "p_sim": 0.0, "ci_sim": (0.0, 0.0),
            "p_near": 0.0, "ci_near": (0.0, 0.0),
            "p_conv": 0.0, "ci_conv": (0.0, 0.0),
            "p_top": 0.0, "ci_top": (0.0, 0.0),
            "p_unr": 0.0, "ci_unr": (0.0, 0.0),
        }

    # Observed statistics (raw)
    n_total = len(pairs)
    n_sim = sum(1 for p in pairs if p["parsed_binary"] == "similar")
    n_near = sum(1 for p in pairs if p["parsed_answer"] == NEAR_PARAPHRASE)
    n_conv = sum(1 for p in pairs if p["parsed_answer"] == CONVERGENT)
    n_top = sum(1 for p in pairs if p["parsed_answer"] == TOPICAL)
    n_unr = sum(1 for p in pairs if p["parsed_answer"] == UNRELATED)

    # RG-corrected point estimates
    raw_sim = n_sim / n_total
    corr_sim = rogan_gladen_correct(raw_sim)
    corr_not_sim = 1.0 - corr_sim
    sim_tot = n_near + n_conv
    np_ratio = n_near / sim_tot if sim_tot > 0 else 0.5
    nsim_tot = n_top + n_unr
    top_ratio = n_top / nsim_tot if nsim_tot > 0 else 0.5

    obs_sim = corr_sim
    obs_near = corr_sim * np_ratio
    obs_conv = corr_sim * (1 - np_ratio)
    obs_top = corr_not_sim * top_ratio
    obs_unr = corr_not_sim * (1 - top_ratio)

    # Bootstrap with resampled sens/spec
    rng = np.random.RandomState(seed)
    boot_sim = np.empty(n_boot)
    boot_near = np.empty(n_boot)
    boot_conv = np.empty(n_boot)
    boot_top = np.empty(n_boot)
    boot_unr = np.empty(n_boot)

    paper_id_arr = np.array(paper_ids)

    for b in range(n_boot):
        sampled = rng.choice(paper_id_arr, size=n_papers, replace=True)
        b_pairs = []
        for pid in sampled:
            b_pairs.extend(by_paper[pid])
        if len(b_pairs) == 0:
            boot_sim[b] = boot_near[b] = boot_conv[b] = 0.0
            boot_top[b] = boot_unr[b] = 0.0
            continue
        n = len(b_pairs)
        b_sens, b_spec = resample_sens_spec(rng)
        b_raw_sim = sum(1 for p in b_pairs if p["parsed_binary"] == "similar") / n
        b_corr_sim = rogan_gladen_correct(b_raw_sim, b_sens, b_spec)
        b_corr_nsim = 1.0 - b_corr_sim
        # Proportional split within similar/not-similar
        b_n_near = sum(1 for p in b_pairs if p["parsed_answer"] == NEAR_PARAPHRASE)
        b_n_conv = sum(1 for p in b_pairs if p["parsed_answer"] == CONVERGENT)
        b_n_top = sum(1 for p in b_pairs if p["parsed_answer"] == TOPICAL)
        b_n_unr = sum(1 for p in b_pairs if p["parsed_answer"] == UNRELATED)
        b_sim_tot = b_n_near + b_n_conv
        b_np_r = b_n_near / b_sim_tot if b_sim_tot > 0 else 0.5
        b_nsim_tot = b_n_top + b_n_unr
        b_top_r = b_n_top / b_nsim_tot if b_nsim_tot > 0 else 0.5

        boot_sim[b] = b_corr_sim
        boot_near[b] = b_corr_sim * b_np_r
        boot_conv[b] = b_corr_sim * (1 - b_np_r)
        boot_top[b] = b_corr_nsim * b_top_r
        boot_unr[b] = b_corr_nsim * (1 - b_top_r)

    return {
        "p_sim": obs_sim,
        "ci_sim": (np.percentile(boot_sim, 2.5), np.percentile(boot_sim, 97.5)),
        "p_near": obs_near,
        "ci_near": (np.percentile(boot_near, 2.5), np.percentile(boot_near, 97.5)),
        "p_conv": obs_conv,
        "ci_conv": (np.percentile(boot_conv, 2.5), np.percentile(boot_conv, 97.5)),
        "p_top": obs_top,
        "ci_top": (np.percentile(boot_top, 2.5), np.percentile(boot_top, 97.5)),
        "p_unr": obs_unr,
        "ci_unr": (np.percentile(boot_unr, 2.5), np.percentile(boot_unr, 97.5)),
    }


# ── LaTeX generation ──────────────────────────────────────────────────
def fmt_pct_ci(val, ci):
    return f"{val*100:.1f}" + r"{\scriptsize\,[" + f"{ci[0]*100:.1f}, {ci[1]*100:.1f}" + "]}"


def generate_table(groups):
    row_order = [
        "H-H (all)",
        "H-H (same reviewer)",
        "H-H (diff reviewer)",
        "A-A (all)",
        "A-A (same model)",
        "A-A (diff model)",
        "H-A",
    ]

    # LaTeX row labels
    row_labels = {
        "H-H (all)": r"H--H (all)",
        "H-H (same reviewer)": r"\quad same reviewer",
        "H-H (diff reviewer)": r"\quad diff reviewer",
        "A-A (all)": r"A--A (all)",
        "A-A (same model)": r"\quad same model",
        "A-A (diff model)": r"\quad diff model",
        "H-A": r"H--A",
    }

    results = {}
    for name in row_order:
        pairs = groups[name]
        print(f"\n--- {name}: {len(pairs)} pairs ---")
        stats = paper_bootstrap(pairs)
        results[name] = (len(pairs), stats)
        print(f"  P(similar) = {stats['p_sim']*100:.1f}%  "
              f"95% CI [{stats['ci_sim'][0]*100:.1f}, {stats['ci_sim'][1]*100:.1f}]")
        print(f"  P(near-para) = {stats['p_near']*100:.1f}%  "
              f"P(convergent) = {stats['p_conv']*100:.1f}%  "
              f"P(topical) = {stats['p_top']*100:.1f}%  "
              f"P(unrelated) = {stats['p_unr']*100:.1f}%")

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\fontsize{7.5}{9.5}\selectfont")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l r cccc c}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Pair type} & \textbf{$N$} "
        r"& \textbf{P(near-para)} & \textbf{P(convergent)} "
        r"& \textbf{P(topical)} & \textbf{P(unrelated)} "
        r"& \textbf{P(similar)} \\"
    )
    lines.append(r"\midrule")

    for name in row_order:
        n_pairs, s = results[name]
        label = row_labels[name]
        cols = [
            label,
            f"{n_pairs:,}",
            fmt_pct_ci(s["p_near"], s["ci_near"]),
            fmt_pct_ci(s["p_conv"], s["ci_conv"]),
            fmt_pct_ci(s["p_top"], s["ci_top"]),
            fmt_pct_ci(s["p_unr"], s["ci_unr"]),
            fmt_pct_ci(s["p_sim"], s["ci_sim"]),
        ]
        lines.append(" & ".join(cols) + r" \\")
        # Add midrule after diff reviewer and diff model blocks
        if name == "H-H (diff reviewer)" or name == "A-A (diff model)":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Review-item-pair similarity across pair types. "
        r"\textit{P(similar)} is the sum of near-paraphrase and convergent-conclusion fractions. "
        r"Within H--H and A--A, rows are further split by whether both items come from the "
        r"same reviewer (or same AI model). "
        r"95\% CIs from 10\,000-iteration paper-level bootstrap.}"
    )
    lines.append(r"\label{tab:similarity-headline}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    return tex


# ── main ───────────────────────────────────────────────────────────────
def main():
    pairs = load_pairs(DATA_FILE)
    groups = group_pairs(pairs)

    for name, ps in groups.items():
        print(f"{name}: {len(ps)} pairs")

    tex = generate_table(groups)

    with open(OUT_TEX, "w") as f:
        f.write(tex)
    print(f"\nWrote {OUT_TEX}")


if __name__ == "__main__":
    main()
