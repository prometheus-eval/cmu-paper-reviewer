# Model Configurations for PeerReview Bench

Hyperparameters used for all AI reviewer agent runs on PeerReview Bench.
Each model runs as an OpenHands v1.5.0 agent with filesystem access to
the paper's source files (preprint, images, code, supplementary).

## Shared Agent Configuration

All models share these OpenHands agent settings:

| Parameter | Value | Notes |
|---|---|---|
| Framework | OpenHands SDK v1.5.0 | |
| Tools | TerminalTool, FileEditorTool, TaskTrackerTool | Agent can read files, run code, and track tasks |
| Max iterations | 5,000 | Per-paper conversation limit |
| Max review items | 5 | Most significant to least significant |
| Criteria preset | Nature | 6 evaluation criteria (Validity, Conclusions, Originality, Data/Methodology, Statistics, Clarity) |
| Condenser | LLMSummarizingCondenser | max_size=200, keep_first=3 |
| Timeout | 600s | Per-API-call timeout |
| Prompt caching | Enabled | `caching_prompt=True` (SDK default) |

## Shared LLM Configuration

Passed to `openhands.sdk.LLM()` for all models:

| Parameter | Value | Notes |
|---|---|---|
| `reasoning_effort` | `"high"` | SDK default; mapped to provider-specific params by LiteLLM |
| `extended_thinking_budget` | `200000` | SDK default; 200K token budget for Anthropic thinking blocks, dropped for non-Anthropic |
| `temperature` | `1.0` | Required by Anthropic extended thinking; applied to all models for consistency |
| `drop_params` | `True` | SDK default; unsupported params silently dropped per provider |
| `num_retries` | `5` | SDK default; retries on transient API errors |

With `drop_params=True`, each provider receives only the parameters it supports:
- **Anthropic models** receive: `reasoning_effort`, `extended_thinking_budget`, `temperature=1.0`
- **OpenAI/GPT models** receive: `reasoning_effort`, `temperature=1.0`
- **Gemini models** receive: `reasoning_effort` (mapped to `thinking_level=high`), `temperature=1.0`
- **Grok, Kimi, Qwen** receive: `temperature=1.0` (built-in reasoning, other params dropped)

## Per-Model Specifications

### GPT Family (via Azure AI)

| | GPT-5.4 | GPT-5.4-mini | GPT-5.2 |
|---|---|---|---|
| Model ID | `azure_ai/gpt-5.4` | `azure_ai/gpt-5.4-mini` | `azure_ai/gpt-5.2` |
| Provider | Azure AI (OpenAI) | Azure AI (OpenAI) | Azure AI (OpenAI) |
| Max input tokens | 1,050,000 | 1,050,000 | 1,050,000 |
| Max output tokens | 128,000 | 128,000 | 128,000 |
| Multimodal (vision) | Yes | Yes | Yes |
| Reasoning | `reasoning_effort="high"` | `reasoning_effort="high"` | `reasoning_effort="high"` |
| Status | New run | New run | Pre-existing reviews |

### Gemini Family (via Google)

| | Gemini 3.1 Pro | Gemini 3 Flash | Gemini 3.0 Pro |
|---|---|---|---|
| Model ID | `gemini/gemini-3.1-pro-preview` | `gemini/gemini-3-flash-preview` | `gemini/gemini-3.0-pro-preview` |
| Provider | Google | Google | Google |
| Max input tokens | 1,048,576 | 1,048,576 | 1,048,576 |
| Max output tokens | 65,536 | 65,535 | 65,536 |
| Multimodal (vision) | Yes | Yes | Yes |
| Reasoning | `thinking_level=high` (via LiteLLM mapping) | `thinking_level=high` | `thinking_level=high` |
| Status | New run | New run | Pre-existing reviews |

### Claude Family (via Anthropic)

| | Claude Opus 4.7 | Claude Opus 4.6 | Claude Opus 4.5 | Claude Sonnet 4.6 |
|---|---|---|---|---|
| Model ID | `anthropic/claude-opus-4-7` | `anthropic/claude-opus-4-6` | `anthropic/claude-opus-4-5` | `anthropic/claude-sonnet-4-6` |
| Provider | Anthropic | Anthropic | Anthropic | Anthropic |
| Max input tokens | 1,000,000 | 1,000,000 | 200,000 | 1,000,000 |
| Max output tokens | 128,000 | 128,000 | 64,000 | 64,000 |
| Multimodal (vision) | Yes | Yes | Yes | Yes |
| Reasoning | Extended thinking (200K budget) | Extended thinking (200K budget) | Extended thinking (200K budget) | Extended thinking (200K budget) |
| Temperature | 1.0 (required for thinking) | 1.0 | 1.0 | 1.0 |
| Status | Pre-existing reviews | — | Pre-existing reviews | New run |

### Other Models

| | Grok 4.1 Fast | Kimi K2.5 | Qwen 3.6 Plus |
|---|---|---|---|
| Model ID | `azure_ai/grok-4-1-fast-reasoning` | `azure_ai/Kimi-K2.5` | `fireworks_ai/.../qwen3p6-plus` |
| Provider | Azure AI (xAI) | Azure AI (Moonshot) | Fireworks AI |
| Max input tokens | 131,072 | 262,144 | ~1,000,000 |
| Max output tokens | 131,072 | 262,144 | 65,536 |
| Multimodal (vision) | Yes | Yes | Yes |
| Reasoning | Built-in (always on) | Built-in (always on) | Always-on reasoning |
| Status | Skipped (tool-use issues) | New run | New run |

## Notes

- **Pre-existing reviews**: GPT-5.2, Gemini 3.0 Pro, and Claude Opus 4.5
  reviews were generated during an earlier phase using the same backend
  reviewer prompt (`reviewer_prompt.py`) but with the production website's
  OpenHands configuration (which may have had slightly different defaults).
  The review filenames in `papers/paper{N}/reviews/` reflect the model
  names at the time of generation.

- **Grok 4.1 Fast Reasoning**: Excluded from the benchmark because the
  model cannot correctly use the OpenHands FileEditorTool's `create`
  command (passes `view_range` instead of `file_text`). The model can
  read and analyze papers but fails to write the review output file.

- **Reviewer prompt**: All models use the same reviewer prompt from
  `backend/reviewer_prompt.py` with the Nature criteria preset (6
  evaluation axes). The prompt instructs the agent to produce at most
  5 review items, sorted from most to least significant.

- **Domain restrictions**: The Tavily search tool is configured to
  exclude results from `nature.com`, `researchsquare.com`,
  `springer.com`, and `springerlink.com` to prevent accessing published
  versions of benchmark papers. This is enforced via both a monkey-patch
  on the MCP tool executor and a prompt instruction.

- **File isolation**: During each review generation, all other models'
  review files, verification code, and trajectories are temporarily
  hidden (renamed with a `._hidden_` suffix) to prevent cross-model
  contamination. Only the current model's own output files are visible.
