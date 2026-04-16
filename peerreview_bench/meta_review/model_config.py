"""
Per-model response_format capabilities + token budgets for LiteLLM routing.

Every model string in MODEL_RESPONSE_FORMAT and MODEL_MAX_OUTPUT_TOKENS is
the bare suffix (without the `litellm_proxy/` prefix). The run script adds
the prefix before calling `litellm.completion`.

The value for each model in MODEL_RESPONSE_FORMAT is one of:
  - "json_schema"  : supports structured outputs via a JSON schema
                     (passed through response_format={"type":"json_schema", ...})
  - "json_object"  : supports free-form JSON output (response_format={"type":"json_object"})
  - None           : no native JSON support — we prompt-engineer the JSON
                     structure and then text-parse the response

MODEL_MAX_OUTPUT_TOKENS is sourced from `litellm.get_model_info(<model>)` —
specifically the `max_output_tokens` field. These values are cached here
so the predictor doesn't need to call `get_model_info` every construction.
If a new model is added, run:

    python3 -c "import litellm; print(litellm.get_model_info('<model>'))"

and paste the max_output_tokens into the dict below.

Source: https://docs.litellm.ai/docs/completion/json_mode plus the LiteLLM
model catalog. If a model you need isn't listed, add it here.
"""

from typing import Dict, Optional

# Keys MUST match the `--model` argument users will pass (bare, no
# litellm_proxy/ prefix). 7 models — all multimodal-capable:
MODEL_RESPONSE_FORMAT: Dict[str, Optional[str]] = {
    # Azure AI
    "azure_ai/gpt-5.4":                       "json_schema",
    "azure_ai/grok-4-1-fast-reasoning":       "json_schema",
    "azure_ai/Kimi-K2.5":                     "json_object",
    # Google / Anthropic
    "gemini/gemini-3.1-pro-preview":          "json_schema",
    "anthropic/claude-opus-4-6":              "json_schema",
    # Fireworks AI — always-on reasoning; json_object is the safe default
    # since this model isn't in the LiteLLM catalog.
    "fireworks_ai/accounts/fireworks/models/qwen3p6-plus":        "json_object",
}


def get_response_format_mode(model_suffix: str) -> Optional[str]:
    """Return 'json_schema' / 'json_object' / None for the given model.

    If the model is unknown, default to 'json_object' (a safe middle ground
    that works on most providers that support any kind of JSON mode).
    """
    if model_suffix.startswith("litellm_proxy/"):
        model_suffix = model_suffix[len("litellm_proxy/"):]
    return MODEL_RESPONSE_FORMAT.get(model_suffix, "json_object")


# Models that accept multimodal (image) input. Used to decide whether to
# attach image blocks to the messages list. Text-only models ignore images.
MULTIMODAL_MODELS = {
    "azure_ai/gpt-5.4",
    "azure_ai/grok-4-1-fast-reasoning",
    "azure_ai/Kimi-K2.5",
    "gemini/gemini-3.1-pro-preview",
    "anthropic/claude-opus-4-6",
    "fireworks_ai/accounts/fireworks/models/qwen3p6-plus",
}


def supports_multimodal(model_suffix: str) -> bool:
    if model_suffix.startswith("litellm_proxy/"):
        model_suffix = model_suffix[len("litellm_proxy/"):]
    return model_suffix in MULTIMODAL_MODELS


# ----------------------------------------------------------------------
# Per-model max output token budgets
# ----------------------------------------------------------------------
#
# Pulled from litellm.get_model_info(<model>)['max_output_tokens']. For the
# models whose catalog entry reports input == output (i.e. a shared context
# budget rather than a separate output cap), we use the reported number
# directly — at runtime, the server will reserve enough for the actual input
# and the remainder becomes the effective output cap.
#
# Max *input* tokens from the same catalog / provider pages, for reference:
#   azure_ai/gpt-5.4                                          1,050,000
#   azure_ai/grok-4-1-fast-reasoning                            131,072
#   azure_ai/Kimi-K2.5                                          262,144
#   gemini/gemini-3.1-pro-preview                              1,048,576
#   anthropic/claude-opus-4-6                                  1,000,000
#   fireworks_ai/.../qwen3p6-plus                              1,000,000
#   fireworks_ai/.../qwen3p6-plus                              1,000,000
#
# All of these comfortably fit the longest paper in the dataset
# (~106K characters ≈ ~26K tokens) plus a review item plus a ~2K system
# prompt, so we never need to truncate input.

MODEL_MAX_OUTPUT_TOKENS: Dict[str, int] = {
    "azure_ai/gpt-5.4":                       128000,
    "azure_ai/grok-4-1-fast-reasoning":       131072,
    "azure_ai/Kimi-K2.5":                     262144,
    "gemini/gemini-3.1-pro-preview":           65536,
    "anthropic/claude-opus-4-6":              128000,
    # Fireworks AI — always-on reasoning; thinking tokens consume the
    # output budget, so set generously.
    "fireworks_ai/accounts/fireworks/models/qwen3p6-plus":     65536,
}

# Fallback if a model isn't in the table above. Large enough that reasoning
# models won't burn their budget on hidden chain-of-thought, small enough
# that we don't wait forever on an unknown model.
DEFAULT_MAX_OUTPUT_TOKENS = 16384


def get_max_output_tokens(model_suffix: str) -> int:
    """Return the per-model default `max_tokens` for a completion call."""
    if model_suffix.startswith("litellm_proxy/"):
        model_suffix = model_suffix[len("litellm_proxy/"):]
    return MODEL_MAX_OUTPUT_TOKENS.get(model_suffix, DEFAULT_MAX_OUTPUT_TOKENS)
