#!/usr/bin/env python3
"""
Smoke test for the 3 similarity_check embedding backends.

Runs each backend on 3 human texts vs 3 AI texts and prints the resulting
pairwise cosine similarity matrix, so you can eyeball that every backend is
reachable and the similarity numbers are sane (diagonal > off-diagonal for
matched pairs).

The qwen3 backend requires `sentence-transformers` + a model download
(~16GB), so we catch the ImportError and skip it if unavailable. The azure
and gemini backends go through the CMU LiteLLM proxy.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent            # .../similarity_check/expert_annotation_similarity
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))       # .../peerreview_bench (for load_data etc)

from embeddings import get_backend  # noqa: E402


BACKENDS = [
    "qwen3",  # local sentence-transformers; may fail if libs missing
    "litellm_proxy/azure_ai/text-embedding-3-large",
    "litellm_proxy/gemini/gemini-embedding-001",
]


# 3 pairs where human[i] and ai[i] SHOULD be about the same topic, so the
# diagonal should dominate in the pairwise matrix.
HUMAN_TEXTS = [
    "The authors did not control for loading rate, which is known to dominate "
    "widget-Y fatigue at the strains they tested. Without varying loading "
    "rate, their reported effect might be an artifact.",
    "The reported p<0.01 result is marginally interesting, but the effect size "
    "d=0.8 is inflated by their sample selection. A pre-registered replication "
    "is essential before drawing strong conclusions.",
    "The schematic in Figure 2 has labels that are too small to read in print. "
    "I suggest increasing the font size on the annotations.",
]

AI_TEXTS = [
    "Loading rate is not controlled in this study. Fatigue behavior of widget-Y "
    "is strongly rate-dependent, so the 10% strain difference may just reflect "
    "the unreported loading-rate variation.",
    "The effect size d=0.8 is surprisingly large for a sample of 200 and p-value "
    "of 0.01. This is suggestive of selection bias; the authors should discuss "
    "whether units were screened before measurement.",
    "Figure 2 text appears illegible at publication size. Please enlarge the "
    "label fonts and use a bolder font weight for the axes.",
]


def _format_matrix(sim: np.ndarray) -> str:
    lines = []
    lines.append("       " + "  ".join(f"ai{j}  " for j in range(sim.shape[1])))
    for i in range(sim.shape[0]):
        row = "  ".join(f"{sim[i,j]:+.3f}" for j in range(sim.shape[1]))
        lines.append(f"  h{i}   {row}")
    return "\n".join(lines)


def main():
    out_dir = _HERE.parent.parent / "outputs" / "similarity_check" / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for backend_name in BACKENDS:
        print("")
        print("=" * 80)
        print(f"Smoke test: {backend_name}")
        print("=" * 80)
        try:
            backend = get_backend(backend_name)
        except Exception as e:
            print(f"  FAILED to instantiate: {type(e).__name__}: {e}")
            summary[backend_name] = {
                "status": "instantiation_failed",
                "error": f"{type(e).__name__}: {e}",
            }
            continue

        print(f"  backend.name: {backend.name}")
        try:
            sim = backend.pairwise(HUMAN_TEXTS, AI_TEXTS)
        except Exception as e:
            traceback.print_exc()
            print(f"  pairwise FAILED: {type(e).__name__}: {e}")
            summary[backend_name] = {
                "status": "pairwise_failed",
                "error": f"{type(e).__name__}: {e}",
            }
            continue

        print(f"  sim.shape: {sim.shape}")
        print(_format_matrix(sim))
        diag = float(np.mean([sim[i, i] for i in range(3)]))
        offdiag = float(
            np.mean([sim[i, j] for i in range(3) for j in range(3) if i != j])
        )
        print(f"  mean_diagonal:    {diag:.3f}")
        print(f"  mean_off_diagonal: {offdiag:.3f}")
        print(f"  margin (higher is better): {diag - offdiag:+.3f}")
        summary[backend_name] = {
            "status": "ok" if diag > offdiag else "ok_weak_signal",
            "backend_display_name": backend.name,
            "sim_matrix": sim.tolist(),
            "mean_diagonal": diag,
            "mean_off_diagonal": offdiag,
            "margin": diag - offdiag,
        }

    (out_dir / "smoke_results.json").write_text(json.dumps(summary, indent=2, default=str))

    print("")
    print("=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    for b, info in summary.items():
        status = info["status"]
        if "mean_diagonal" in info:
            tag = f"diag={info['mean_diagonal']:.3f}  off={info['mean_off_diagonal']:.3f}  margin={info['margin']:+.3f}"
        else:
            tag = info.get("error", "")
        print(f"  {b:<60}  {status:<20}  {tag}")
    print("")
    print(f"Full results: {out_dir / 'smoke_results.json'}")


if __name__ == "__main__":
    main()
