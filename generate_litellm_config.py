#!/usr/bin/env python3
"""Generate LiteLLM Proxy config + vLLM thinking-toggle rules from capability JSON.

Requires: PyYAML (`pip install pyyaml`). Run in a directory containing
thinking_hook.py and thinking_transform.py (included with this distribution).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


# Non-chat endpoint mappings are only routing hints, not proof that the vLLM
# deployment exposes the mapped endpoint.
NON_CHAT_ENDPOINTS: dict[str, tuple[str, str]] = {
    "embedding": ("embedding", "/v1/embeddings"),
    "image_generation": ("image_generation", "/v1/images/generations"),
    "speech_to_text": ("audio_transcription", "/v1/audio/transcriptions"),
    "text_to_speech": ("audio_speech", "/v1/audio/speech"),
}
BUDGET_SENTINEL = "positive_integer_budget"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def get_toggle_rule(model_id: str, meta: dict[str, Any], budget_on: int) -> dict[str, Any] | None:
    """Extract *toggle-only* models; leave native-effort models untouched.

    Unknown effort options (null) are permitted only if the native effort
    parameter is absent. A model with effort options configured is not a
    toggle-only model.
    """
    if meta.get("chat_supported") is not True:
        return None
    thinking = meta.get("thinking") or {}
    if thinking.get("supported") is not True:
        return None
    toggle = thinking.get("toggle")
    if not isinstance(toggle, dict):
        return None
    effort = meta.get("reasoning_effort") or {}
    if effort.get("parameter") or effort.get("options"):
        return None

    parameter = toggle.get("parameter")
    if not isinstance(parameter, str) or not parameter.strip():
        raise ValueError(f"{model_id}: missing toggle.parameter")
    if "on" not in toggle or "off" not in toggle:
        raise ValueError(f"{model_id}: missing toggle.on or toggle.off")
    on_value = toggle["on"]
    if on_value == BUDGET_SENTINEL:
        on_value = budget_on
    off_value = toggle["off"]
    if on_value is None or off_value is None:
        raise ValueError(f"{model_id}: toggle on/off cannot be null")
    if on_value == off_value:
        raise ValueError(f"{model_id}: toggle on/off must differ")
    if not all(part for part in parameter.split(".")):
        raise ValueError(f"{model_id}: invalid toggle parameter")
    # Accept either `chat_template_kwargs.enable_thinking` or `extra_body.x.y`.
    return {"parameter": parameter, "on": on_value, "off": off_value}


def resolve_endpoint(model_id: str, endpoint_map: dict[str, Any], default_base: str, default_key: str, provider: str) -> dict[str, Any]:
    item = endpoint_map.get(model_id, {})
    if isinstance(item, str):
        item = {"api_base": item}
    if not isinstance(item, dict):
        raise ValueError(f"{model_id}: endpoint map value must be a string or object")
    for field in item:
        if field not in {"api_base", "api_key", "served_model_name", "provider"}:
            raise ValueError(f"{model_id}: unknown endpoint setting: {field}")
    p = item.get("provider", provider)
    if p not in {"openai", "hosted_vllm"}:
        raise ValueError(f"{model_id}: invalid provider {p!r}")
    base = item.get("api_base", default_base)
    served = item.get("served_model_name", model_id)
    api_key = item.get("api_key", default_key)
    if not isinstance(base, str) or not base.strip():
        raise ValueError(f"{model_id}: missing vLLM api_base")
    if not isinstance(served, str) or not served.strip():
        raise ValueError(f"{model_id}: missing served_model_name")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"{model_id}: missing api_key (use 'anything' for unauthenticated vLLM)")
    if api_key.startswith("os.environ/"):
        raise ValueError(f"{model_id}: api_key must be a literal value, not an environment reference")
    return {"model": f"{p}/{served}", "api_base": base, "api_key": api_key}


def build(capabilities: dict[str, Any], endpoint_map: dict[str, Any], *,
          api_base: str, api_key: str, provider: str,
          include_non_chat: bool, include_unknown: bool,
          thinking_budget_on: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    models = capabilities.get("models")
    if not isinstance(models, dict):
        raise ValueError("Input must contain a 'models' object")
    if thinking_budget_on <= 0:
        raise ValueError("--thinking-budget-on must be a positive integer")
    if not isinstance(endpoint_map, dict):
        raise ValueError("endpoint map must be a JSON object")
    unknown_ids = set(endpoint_map) - set(models)
    if unknown_ids:
        raise ValueError(f"Unknown model(s) in endpoint map: {', '.join(sorted(unknown_ids))}")

    if api_key.startswith("os.environ/"):
        raise ValueError("--api-key must be literal, not os.environ/...; supply the actual key")

    deployments: list[dict[str, Any]] = []
    rules: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for model_id, meta in models.items():
        if not isinstance(meta, dict):
            raise ValueError(f"{model_id}: expected capability object")
        chat = meta.get("chat_supported")
        modality = meta.get("modality", "unknown")
        if chat is True:
            model_info = {"mode": "chat", "supported_endpoints": ["/v1/chat/completions"]}
        elif chat is False:
            if not include_non_chat:
                skipped[model_id] = "non_chat (use --include-non-chat to include supported modalities)"
                continue
            target = NON_CHAT_ENDPOINTS.get(modality)
            if target is None:
                skipped[model_id] = "non_chat endpoint unknown; explicit routing required"
                continue
            mode, endpoint = target
            model_info = {
                "mode": mode,
                "supported_endpoints": [endpoint],
                "enforce_supported_endpoints": True,
            }
        elif chat is None:
            if not include_unknown:
                skipped[model_id] = "chat capability unknown (use --include-unknown)"
                continue
            model_info = {"mode": "chat", "supported_endpoints": ["/v1/chat/completions"]}
        else:
            raise ValueError(f"{model_id}: chat_supported must be boolean or null")

        # LiteLLM model_info describes whether reasoning can be requested by
        # clients. A toggle-only model exposes only `medium`; the app uses
        # omission/null for OFF. Native effort models expose their own levels.
        effort = meta.get("reasoning_effort") or {}
        thinking = meta.get("thinking") or {}
        rule = get_toggle_rule(model_id, meta, thinking_budget_on)
        native_levels = effort.get("options")
        if native_levels is not None and (not isinstance(native_levels, list) or not all(isinstance(x, str) for x in native_levels)):
            raise ValueError(f"{model_id}: reasoning_effort.options must be string list or null")
        if rule is not None:
            model_info["supports_reasoning"] = True
            model_info["reasoning_effort_levels"] = ["medium"]
        elif chat is True and (thinking.get("supported") is True or effort.get("parameter") or native_levels):
            model_info["supports_reasoning"] = True
            model_info["reasoning_effort_levels"] = native_levels if native_levels is not None else []
        else:
            model_info["supports_reasoning"] = False
            model_info["reasoning_effort_levels"] = []

        upstream = resolve_endpoint(model_id, endpoint_map, api_base, api_key, provider)
        deployments.append({
            "model_name": model_id,
            "litellm_params": upstream,
            "model_info": model_info,
        })
        if rule is not None:
            rules[model_id] = rule

    config = {
        "model_list": deployments,
        "litellm_settings": {"callbacks": "thinking_hook.thinking_hook"},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }
    mapping = {"schema_version": 1, "toggle_only_models": rules}
    report = {
        "total_input_models": len(models),
        "included_models": len(deployments),
        "hooked_toggle_only_models": sorted(rules),
        "skipped_models": skipped,
        "note": "Endpoint support and toggle metadata are not verified against running vLLM servers.",
    }
    return config, mapping, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="vllm_model_capabilities.json")
    p.add_argument("--output-dir", type=Path, default=Path("generated"))
    p.add_argument("--api-base", default="os.environ/VLLM_API_BASE", help="Common vLLM API base URL or env reference")
    p.add_argument("--api-key", default="anything", help="Literal vLLM API key written directly into YAML (default: anything)")
    p.add_argument("--provider", choices=["openai", "hosted_vllm"], default="openai")
    p.add_argument("--endpoint-map", type=Path, help="JSON per-model api_base, served_model_name, api_key, provider overrides")
    p.add_argument("--include-non-chat", action="store_true", help="Include supported audio/image/embedding modalities")
    p.add_argument("--include-unknown", action="store_true", help="Include models with chat_supported=null as chat (requires verification)")
    p.add_argument("--thinking-budget-on", type=int, default=2048, help="ON value for 'positive_integer_budget' sentinel")
    args = p.parse_args()

    capabilities = load_object(args.input)
    endpoint_map = load_object(args.endpoint_map) if args.endpoint_map else {}
    config, rules, report = build(
        capabilities, endpoint_map,
        api_base=args.api_base, api_key=args.api_key, provider=args.provider,
        include_non_chat=args.include_non_chat, include_unknown=args.include_unknown,
        thinking_budget_on=args.thinking_budget_on,
    )
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "thinking_rules.json").write_text(json.dumps(rules, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "generation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in ("thinking_hook.py", "thinking_transform.py"):
        src = Path(__file__).resolve().with_name(name)
        dst = out / name
        if src != dst:
            shutil.copy2(src, dst)
    print(f"Generated {len(config['model_list'])} deployments and {len(rules['toggle_only_models'])} thinking rules in {out}")
    for model, reason in report["skipped_models"].items():
        print(f"  skip {model}: {reason}")


if __name__ == "__main__":
    main()
