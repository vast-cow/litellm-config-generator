"""Pure Python transform for LiteLLM's async_pre_call_hook.

App-side protocol for toggle-only vLLM models:
    reasoning_effort omitted / null => thinking OFF
    reasoning_effort = 'medium'  => thinking ON
No other value (including 'middle') is accepted.

For models with native reasoning_effort support there is no rewrite.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class ThinkingRequestError(ValueError):
    """Bad client reasoning_effort or invalid nested request data."""


_MISSING = object()


def _get_request_effort(data: dict[str, Any]) -> Any:
    top = data.get("reasoning_effort", _MISSING)
    extra = data.get("extra_body", _MISSING)
    nested = extra.get("reasoning_effort", _MISSING) if isinstance(extra, dict) else _MISSING
    if top is not _MISSING and nested is not _MISSING and top != nested:
        raise ThinkingRequestError("reasoning_effort specified with conflicting values")
    return top if top is not _MISSING else (nested if nested is not _MISSING else None)


def _set_nested(root: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    if not keys or not all(k and k.strip() for k in keys):
        raise ThinkingRequestError(f"Invalid toggle parameter: {dotted!r}")
    current = root
    for key in keys[:-1]:
        child = current.setdefault(key, {})
        if not isinstance(child, dict):
            raise ThinkingRequestError(f"Expected object at {key!r} in toggle parameter")
        current = child
    current[keys[-1]] = value


def rewrite_request(data: dict[str, Any], rules: dict[str, Any], call_type: str) -> dict[str, Any]:
    """Return rewritten request, without mutating input; raise ThinkingRequestError on invalid effort."""
    if call_type != "completion":
        return data
    model = data.get("model")
    rule = rules.get(model)
    if rule is None:
        return data

    result = deepcopy(data)
    effort = _get_request_effort(result)
    if effort is None:
        toggle = rule["off"]
    elif effort == "medium":
        toggle = rule["on"]
    else:
        raise ThinkingRequestError(f"Model {model!r} only accepts reasoning_effort=null or 'medium'; got {effort!r}")

    # Remove application-level reasoning_effort so the upstream vLLM never
    # sees the virtual switch value in addition to the model-native toggle.
    result.pop("reasoning_effort", None)
    extra = result.setdefault("extra_body", {})
    if not isinstance(extra, dict):
        raise ThinkingRequestError("extra_body must be a JSON object")
    extra.pop("reasoning_effort", None)

    parameter = rule["parameter"]
    if parameter.startswith("extra_body."):
        parameter = parameter[len("extra_body."):]
    # If metadata explicitly says to write a top-level request field, honor it.
    if parameter == "reasoning_effort":
        result["reasoning_effort"] = toggle
    else:
        # vLLM-specific parameters must be sent via extra_body, so they are
        # flattened by LiteLLM's OpenAI-compatible provider on the wire.
        _set_nested(extra, parameter, toggle)

    return result
