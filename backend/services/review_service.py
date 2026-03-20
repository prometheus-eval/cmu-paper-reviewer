"""Review service refactored from generate_review.py — orchestrates the OpenHands agent."""

import json
import logging
import os
import random
import uuid

import litellm
from openhands.sdk import Agent, Conversation, Event, LLM, LLMConvertibleEvent, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal.definition import TerminalTool

from backend.config import settings
from backend.reviewer_prompt import build_reviewer_prompt
from backend.services.storage_service import preprint_dir, review_md_path, review_output_dir

logger = logging.getLogger(__name__)


class ReviewService:
    def __init__(
        self,
        litellm_api_key: str | None = None,
        litellm_base_url: str | None = None,
        tavily_api_key: str | None = None,
        review_settings: dict | None = None,
    ):
        # Randomly pick a model from the configured list
        self.model_name = random.choice(settings.review_models)
        self.litellm_api_key = litellm_api_key or settings.litellm_api_key
        self.litellm_base_url = litellm_base_url or settings.litellm_base_url
        self.tavily_api_key = tavily_api_key or settings.tavily_api_key
        self.review_settings = review_settings

    def _build_llm(self) -> LLM:
        # Disable Claude/Anthropic-specific params that non-Claude models
        # (e.g. Azure AI GPT-5.4, Gemini) don't support.
        is_claude = "claude" in self.model_name.lower()

        # For non-Claude models behind a LiteLLM proxy, the proxy may reject
        # params like tool_choice that aren't in its allowed list.
        # Passing allowed_openai_params via extra_body whitelists them per-request.
        extra_body = {}
        if not is_claude:
            extra_body["allowed_openai_params"] = ["tool_choice"]

        return LLM(
            model=self.model_name,
            base_url=self.litellm_base_url,
            api_key=self.litellm_api_key,
            drop_params=True,
            prompt_cache_retention="24h" if is_claude else None,
            caching_prompt=is_claude,
            reasoning_effort="high" if is_claude else None,
            extended_thinking_budget=200000 if is_claude else None,
            enable_encrypted_reasoning=is_claude,
            litellm_extra_body=extra_body,
        )

    def _build_mcp_config(self) -> dict:
        if not self.tavily_api_key:
            return {}

        import sys

        # Always use our custom MCP server (more reliable than npx mcp-remote
        # which uses a fragile SSE proxy to mcp.tavily.com).
        args = [
            sys.executable, "-m", "backend.services.tavily_mcp",
            "--api-key", self.tavily_api_key,
        ]

        # Add date filtering if user disabled future references and we have a paper date
        if self.review_settings:
            enable_future = self.review_settings.get("enable_future_references", True)
            paper_date = self.review_settings.get("paper_date")
            if not enable_future and paper_date:
                args.extend(["--paper-date", paper_date])

        return {
            "tavily": {
                "command": args[0],
                "args": args[1:],
            }
        }

    @staticmethod
    def _patch_litellm_for_proxy():
        """Monkey-patch litellm completion to inject allowed_openai_params for proxy models.

        The LiteLLM proxy rejects tool_choice for azure_ai models unless
        allowed_openai_params is sent in extra_body. We must patch every
        module that imports litellm's completion/acompletion functions.
        """
        if getattr(litellm, "_cmu_patched", False):
            return  # Already patched

        # We need to inject extra_body={'allowed_openai_params': ['tool_choice']}
        # into every litellm call that goes through the proxy. The proxy rejects
        # tool_choice unless explicitly whitelisted per-request.

        _original_completion = litellm.completion
        _original_acompletion = litellm.acompletion

        def _inject_allowed_params(kwargs):
            """Inject allowed_openai_params into extra_body for proxy models."""
            model = kwargs.get("model", "")
            if "litellm_proxy/" in str(model) or "azure_ai" in str(model) or "gemini" in str(model):
                extra_body = kwargs.get("extra_body") or {}
                extra_body["allowed_openai_params"] = ["tool_choice"]
                kwargs["extra_body"] = extra_body
                logger.debug("Injected allowed_openai_params for model=%s", model)

        def _patched_completion(*args, **kwargs):
            _inject_allowed_params(kwargs)
            return _original_completion(*args, **kwargs)

        async def _patched_acompletion(*args, **kwargs):
            _inject_allowed_params(kwargs)
            return await _original_acompletion(*args, **kwargs)

        # 1. Patch the litellm module itself
        litellm.completion = _patched_completion
        litellm.acompletion = _patched_acompletion

        # 2. Patch ALL already-imported references across every loaded module
        import sys
        patched_modules = []
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            try:
                if getattr(mod, "litellm_completion", None) is _original_completion:
                    mod.litellm_completion = _patched_completion
                    patched_modules.append(f"{mod_name}.litellm_completion")
                if getattr(mod, "litellm_acompletion", None) is _original_acompletion:
                    mod.litellm_acompletion = _patched_acompletion
                    patched_modules.append(f"{mod_name}.litellm_acompletion")
                # Also check for plain 'completion' attr that IS litellm's
                if mod_name != "litellm" and getattr(mod, "completion", None) is _original_completion:
                    mod.completion = _patched_completion
                    patched_modules.append(f"{mod_name}.completion")
            except Exception:
                pass

        # 3. Install an import hook so any FUTURE imports also get patched
        import importlib.abc
        import importlib.machinery

        class _LiteLLMPatchFinder(importlib.abc.MetaPathFinder):
            def find_module(self, fullname, path=None):
                return self if "openhands" in fullname else None

            def load_module(self, fullname):
                if fullname in sys.modules:
                    mod = sys.modules[fullname]
                    if getattr(mod, "litellm_completion", None) is _original_completion:
                        mod.litellm_completion = _patched_completion
                    if getattr(mod, "litellm_acompletion", None) is _original_acompletion:
                        mod.litellm_acompletion = _patched_acompletion
                    return mod
                return None

        sys.meta_path.insert(0, _LiteLLMPatchFinder())

        litellm._cmu_patched = True
        logger.info("Patched litellm for proxy (allowed_openai_params). Modules: %s", patched_modules or "(litellm only)")

    def run_review(self, key: str) -> tuple[str, str]:
        """Run the OpenHands review agent for a given submission.

        Returns a tuple of (path to review markdown, model name used).
        """
        logger.info("Starting review for key=%s with model=%s", key, self.model_name)

        # Patch litellm to inject allowed_openai_params for proxy models
        litellm.drop_params = True
        self._patch_litellm_for_proxy()

        llm = self._build_llm()
        condenser = LLMSummarizingCondenser(
            llm=llm.model_copy(update={"usage_id": "condenser"}),
            max_size=200,
            keep_first=3,
        )

        agent = Agent(
            llm=llm,
            tools=[
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
                Tool(name=TaskTrackerTool.name),
            ],
            mcp_config=self._build_mcp_config(),
            condenser=condenser,
        )

        link_to_paper = str(preprint_dir(key))
        model_short = self.model_name.split("/")[-1]
        readable_id = f"{self.model_name.replace('/', '_')}_{key}".replace(".", "_").replace("-", "_")
        conversation_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, readable_id)

        # Ensure review output directory exists
        review_output_dir(key).mkdir(parents=True, exist_ok=True)

        cwd = os.getcwd()
        conversation = Conversation(
            agent=agent,
            workspace=cwd,
            persistence_dir=str(review_output_dir(key) / f"{model_short}_trajectory"),
            conversation_id=conversation_uuid,
            max_iteration_per_run=200,
        )

        prompt = build_reviewer_prompt(self.review_settings)
        prompt = prompt.replace("[LINK TO THE PAPER]", link_to_paper).replace(
            "[MODEL NAME]", model_short
        )
        conversation.send_message(prompt)
        conversation.run()

        cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
        logger.info("Review complete for key=%s, cost=%s", key, cost)

        del conversation

        # The agent writes to review_[MODEL NAME].md — copy/rename to review.md
        agent_review_path = review_output_dir(key) / f"review_{model_short}.md"
        canonical_path = review_md_path(key)
        if agent_review_path.exists() and not canonical_path.exists():
            agent_review_path.rename(canonical_path)
        elif agent_review_path.exists():
            # If review.md already exists, overwrite
            canonical_path.write_text(agent_review_path.read_text(encoding="utf-8"), encoding="utf-8")

        return str(canonical_path), self.model_name
