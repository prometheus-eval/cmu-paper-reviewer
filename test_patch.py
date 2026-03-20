from backend.services.review_service import _patched_completion
import openhands.sdk.llm.llm as sdk
import litellm
assert sdk.litellm_completion is _patched_completion, "SDK NOT PATCHED"
assert litellm.completion is _patched_completion, "litellm NOT PATCHED"
assert litellm.main.completion is _patched_completion, "litellm.main NOT PATCHED"
print("ALL PATCHES VERIFIED")
