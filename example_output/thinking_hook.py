"""LiteLLM Proxy pre-call hook for per-model vLLM thinking toggles.

Register this *instance* with: litellm_settings.callbacks: thinking_hook.thinking_hook
Place thinking_rules.json in the same directory, or override THINKING_RULES_PATH.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger

from thinking_transform import ThinkingRequestError, rewrite_request


class ThinkingToggleHook(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        path = Path(os.environ.get("THINKING_RULES_PATH", Path(__file__).with_name("thinking_rules.json")))
        rules_doc = json.loads(path.read_text(encoding="utf-8"))
        self.rules = rules_doc["toggle_only_models"]
        if not isinstance(self.rules, dict):
            raise ValueError("thinking_rules.json must contain toggle_only_models object")

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            return rewrite_request(data, self.rules, call_type)
        except ThinkingRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


thinking_hook = ThinkingToggleHook()
