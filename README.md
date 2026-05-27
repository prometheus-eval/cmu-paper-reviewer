<p align="center">
  <img src="https://raw.githubusercontent.com/prometheus-eval/prometheus-eval/main/assets/logo.png" alt="Prometheus-Logo" style="width: 15%; display: block; margin: auto;">
</p>

<h1 align="center">CMU Paper Reviewer</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.20668"><img src="https://img.shields.io/badge/arXiv-2605.20668-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/prometheus-eval/peerreview-bench"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-ff9d00" alt="Dataset"></a>
  <a href="https://prometheus-eval.github.io/cmu-paper-reviewer/"><img src="https://img.shields.io/badge/Website-CMU%20Paper%20Reviewer-blue" alt="Website"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
</p>

<p align="center">
  Upload your manuscript and get feedback on the five most critical issues to address. <br>
</p>

---

## Quick Start

The easiest way to use our system is to submit your manuscript through our [website](https://prometheus-eval.github.io/cmu-paper-reviewer/). We offer 3 free trials per day! We also support other ways to use our system, depending on your needs:

### 1. I want to receive more than 3 reviews per day

You can receive more than 3 reviews by using your own API keys on our [website](https://prometheus-eval.github.io/cmu-paper-reviewer/). You'll need to prepare the following API keys:

- **Mistral API** — used for OCR. You can sign up for an API key through this [link](https://console.mistral.ai/home).
- **LiteLLM API** — used for running the agent via LiteLLM. Please read this [page](https://docs.litellm.ai/docs/proxy/quick_start) and prepare a key from any LiteLLM-compatible provider.
    - If you aren't sure which provider to use, [OpenRouter](https://openrouter.ai/) is most likely your best choice. In this case, set your base URL to `https://openrouter.ai/api/v1`.
- **Tavily API** — used by the agent to search for literature while writing its review. Empirically, we found this to be crucial for generating high-quality reviews. You can sign up for an API key through this [link](https://www.tavily.com/).

### 2. I'm a researcher/developer and would like to experiment with your harness/system

We open-sourced our code for research purposes! Please read more about our system below. If you have any specific questions, feel free to open an issue or a PR.

If you're developing your own AI reviewer, consider using our [PeerReview Bench](https://huggingface.co/datasets/prometheus-eval/peerreview-bench) to evaluate how well it generates reviews for scientific papers. This is especially useful if your reviewer was trained mainly on AI papers — evaluating it on non-AI scientific papers makes for a good out-of-domain testbed.

### 3. I'm a conference or journal organizer and would like to deploy your service

Please send an email to seungone@cmu.edu and we can discuss!

---

## Evaluating your own AI reviewer with PeerReview Bench

If you're building an AI reviewer, [**PeerReview Bench**](https://huggingface.co/datasets/prometheus-eval/peerreview-bench) measures how well it reviews real scientific papers along two axes:

- **Recall** — how many of the important, expert-validated issues your reviewer catches.
- **Precision** — how good your reviewer's own items are, judged by an LLM meta-reviewer on correctness, significance, and evidence.

The harness lives in [`peerreview_bench/`](peerreview_bench/) and downloads all papers and human review items from HuggingFace — no private data required.

```bash
pip install -r peerreview_bench/requirements.txt
export LITELLM_API_KEY=<your-key>
export LITELLM_BASE_URL=https://cmu.litellm.ai   # or your own LiteLLM endpoint

cd peerreview_bench/evaluation
python3 prepare_papers.py                              # 1. download papers + rubric (one-time)
python3 generate_reviews.py --limit 5                  # 2a. run your agent, or...
#   ...2b. bring your own: drop review_items_<model>.json into each papers/paper{N}/review/
python3 evaluate.py --limit 5                          # 3. score recall + precision (F1)
```

See [`peerreview_bench/README.md`](peerreview_bench/README.md) for the bring-your-own-judge (BYOJ) format, configurable parameters, and cost estimates.

---

## Experimenting with the CMU Paper Reviewer

### How it works

The reviewer is an agentic pipeline, served by a FastAPI backend with a background worker and a static frontend (deployable to GitHub Pages via `docs/`):

```
PDF upload → Mistral OCR → OpenHands agent (GPT-5.5) → Markdown review → LaTeX PDF
```

The core is the **OpenHands agent** in `backend/services/review_service.py`. It runs with file-editor, terminal, and task-tracker tools, plus a Tavily MCP server so it can search the literature while reviewing — empirically the single biggest driver of review quality. The agent reads the OCR'd manuscript (and any attached code/supplementary materials), and writes a Markdown review of the most critical issues.

### What you can modify

Most experimentation happens in three files:

- **`backend/reviewer_prompt.py`** — the review rubric and prompt. Swap between the built-in `nature` and `neurips` criteria presets (or define `custom` ones), cap the number of items (`max_items`, default 5), and toggle behaviors like literature search (`enable_future_references`) and limitation-focused critique (`criticize_limitations`).
- **`backend/config.py`** — the LLM and OCR models (the agent picks randomly from `review_models`), the LiteLLM endpoint, page/size limits, and rate limiting.
- **`backend/services/review_service.py`** — the agent's tools and LLM parameters (reasoning effort, extended thinking budget, prompt caching, encrypted reasoning for stateless providers).

Two reviewer features are worth knowing when experimenting:

- **Optional uploads** — submitters can attach code (`.zip`, extracted to `preprint/code/`) and supplementary materials (`.pdf`, saved to `preprint/supplementary/`) for the agent to inspect during review.
- **Verification code** — the agent may write scripts to reproduce a paper's claims; anything it leaves under `verification_code_*` is surfaced back to the user.

### Running it locally

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your MISTRAL / LITELLM / TAVILY keys
```

```bash
uvicorn backend.main:app --reload     # 1. API server
python -m backend.worker              # 2. background worker (separate terminal)
python -m http.server 5500 --directory docs   # 3. frontend
```

### Project layout

```
backend/
├── main.py              # FastAPI app
├── config.py            # Settings from env vars
├── database.py          # SQLAlchemy async + SQLite
├── models.py            # Submission + Annotation ORM models
├── schemas.py           # Pydantic request/response schemas
├── reviewer_prompt.py   # Review agent prompt / rubric
├── worker.py            # Background job processor
├── routers/             # /api/submit, /api/status, /api/review endpoints
└── services/
    ├── ocr_service.py             # Mistral OCR
    ├── review_service.py          # OpenHands agent orchestration
    ├── tavily_mcp.py              # Tavily MCP server (date-filtered literature search)
    ├── paper_date_service.py      # extract the paper's date (OCR text / OpenAlex)
    ├── reference_date_service.py  # tag review citations as [BEFORE]/[AFTER] the paper
    ├── pdf_service.py             # LaTeX PDF generation (weasyprint fallback)
    ├── email_service.py           # HTML email notifications
    └── storage_service.py         # File path management

docs/                    # GitHub Pages frontend (upload + review pages)
peerreview_bench/        # the evaluation benchmark (see section above)
```

## Deployment

- **Frontend** — push to GitHub, enable GitHub Pages from the `docs/` folder.
- **Backend** — deploy to any server (VPS, cloud); update `API_BASE_URL` in `docs/js/config.js` and `CORS_ORIGINS` in `.env`.

## Citation

If this was useful, please consider citing our publication!

```bibtex
@article{kim2026limits,
  title={On the limits and opportunities of AI reviewers: Reviewing the reviews of Nature-family papers with 45 expert scientists},
  author={Kim, Seungone and Yoon, Dongkeun and Gashteovski, Kiril and Suk, Juyoung and Baek, Jinheon and Aggarwal, Pranjal and Wu, Ian and Zaverkin, Viktor and Petkoski, Spase and Schrider, Daniel R and others},
  journal={arXiv preprint arXiv:2605.20668},
  year={2026}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
