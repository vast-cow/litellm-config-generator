import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_litellm_config import build, get_toggle_rule
from thinking_transform import ThinkingRequestError, rewrite_request

QWEN = "Qwen/Qwen3.6-35B-A3B"
SEED = "ByteDance-Seed/Seed-OSS-36B-Instruct"
DEEPSEEK = "deepseek-ai/DeepSeek-V4-Flash"
EMBEDDING = "RiVault/Embedding-Small"
GEMMA = "google/gemma-4-26B-A4B-it"


def fixture():
    return {"models": {
        QWEN: {"chat_supported": True, "modality": "text_chat", "thinking": {"supported": True, "toggle": {"parameter": "chat_template_kwargs.enable_thinking", "on": True, "off": False}}, "reasoning_effort": {"parameter": None, "options": None}},
        GEMMA: {"chat_supported": True, "modality": "text_chat", "thinking": {"supported": True, "toggle": {"parameter": "chat_template_kwargs.enable_thinking", "on": True, "off": False}}, "reasoning_effort": {"parameter": None, "options": []}},
        SEED: {"chat_supported": True, "modality": "text_chat", "thinking": {"supported": True, "toggle": {"parameter": "chat_template_kwargs.thinking_budget", "on": "positive_integer_budget", "off": 0}}, "reasoning_effort": {"parameter": None, "options": []}},
        DEEPSEEK: {"chat_supported": True, "modality": "text_chat", "thinking": {"supported": True, "toggle": {"parameter": "reasoning_effort", "on": "high", "off": "none"}}, "reasoning_effort": {"parameter": "reasoning_effort", "options": ["low", "high", "max"]}},
        EMBEDDING: {"chat_supported": False, "modality": "embedding", "thinking": {"supported": False, "toggle": None}, "reasoning_effort": {"parameter": None, "options": []}},
    }}


def build_fixture(**kwargs):
    return build(fixture(), {}, api_base="http://localhost:8000/v1", api_key="my-direct-api-key", provider="openai", include_non_chat=False, include_unknown=False, thinking_budget_on=2048, **kwargs)


def test_generation_and_toggle_only_detection():
    config, mapping, report = build_fixture()
    assert len(config["model_list"]) == 4
    assert config["litellm_settings"]["callbacks"] == "thinking_hook.thinking_hook"
    assert config["model_list"][0]["litellm_params"]["model"] == "openai/" + QWEN
    assert set(mapping["toggle_only_models"]) == {QWEN, SEED, GEMMA}
    assert mapping["toggle_only_models"][SEED]["on"] == 2048
    assert EMBEDDING in report["skipped_models"]


def test_none_and_missing_are_off():
    rule = get_toggle_rule(QWEN, fixture()["models"][QWEN], 2048)
    for payload in ({"model": QWEN}, {"model": QWEN, "reasoning_effort": None}, {"model": QWEN, "extra_body": {"reasoning_effort": None}}):
        out = rewrite_request(payload, {QWEN: rule}, "completion")
        assert out["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert "reasoning_effort" not in out
        assert "reasoning_effort" not in out["extra_body"]
        assert payload.get("reasoning_effort") is None  # input unchanged


def test_medium_only_is_on_and_preserves_other_kwargs():
    rule = get_toggle_rule(QWEN, fixture()["models"][QWEN], 2048)
    original = {"model": QWEN, "reasoning_effort": "medium", "extra_body": {"chat_template_kwargs": {"foo": 123, "enable_thinking": False}, "other": "ok"}}
    result = rewrite_request(original, {QWEN: rule}, "completion")
    assert result["extra_body"]["chat_template_kwargs"] == {"foo": 123, "enable_thinking": True}
    assert result["extra_body"]["other"] == "ok"
    assert "reasoning_effort" not in result
    assert original["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("invalid", ["middle", "low", "high", "xhigh", 1, False, "none", "MEDIUM"])
def test_invalid_efforts_rejected(invalid):
    rule = get_toggle_rule(QWEN, fixture()["models"][QWEN], 2048)
    with pytest.raises(ThinkingRequestError):
        rewrite_request({"model": QWEN, "reasoning_effort": invalid}, {QWEN: rule}, "completion")


def test_seed_budget_and_nested_effort():
    rule = get_toggle_rule(SEED, fixture()["models"][SEED], 8192)
    req = {"model": SEED, "extra_body": {"reasoning_effort": "medium"}}
    out = rewrite_request(req, {SEED: rule}, "completion")
    assert out["extra_body"] == {"chat_template_kwargs": {"thinking_budget": 8192}}
    req2 = {"model": SEED}
    assert rewrite_request(req2, {SEED: rule}, "completion")["extra_body"]["chat_template_kwargs"]["thinking_budget"] == 0


def test_native_effort_unmodified_and_other_call_types_pass_through():
    config, mapping, report = build_fixture()
    src = {"model": DEEPSEEK, "reasoning_effort": "max"}
    assert rewrite_request(src, mapping["toggle_only_models"], "completion") is src
    src2 = {"model": QWEN, "reasoning_effort": "low"}
    assert rewrite_request(src2, mapping["toggle_only_models"], "embeddings") is src2


def test_conflicting_effort_values_rejected():
    rule = get_toggle_rule(QWEN, fixture()["models"][QWEN], 2048)
    with pytest.raises(ThinkingRequestError, match="conflicting"):
        rewrite_request({"model": QWEN, "reasoning_effort": "medium", "extra_body": {"reasoning_effort": None}}, {QWEN: rule}, "completion")


def test_include_embedding_with_endpoint_guard():
    d = fixture()
    config, mapping, report = build(d, {}, api_base="http://localhost:8000/v1", api_key="envkey", provider="hosted_vllm", include_non_chat=True, include_unknown=False, thinking_budget_on=2048)
    byname = {i["model_name"]: i for i in config["model_list"]}
    assert byname[EMBEDDING]["model_info"]["supported_endpoints"] == ["/v1/embeddings"]
    assert byname[EMBEDDING]["model_info"]["enforce_supported_endpoints"] is True
    assert byname[EMBEDDING]["litellm_params"]["model"] == "hosted_vllm/"+EMBEDDING


def test_api_endpoints_with_per_model_served_names():
    config, mapping, report = build(fixture(), {QWEN: {"api_base": "http://gpu1:8000/v1", "served_model_name": "qwen"}}, api_base="http://gpu0:8000/v1", api_key="key", provider="openai", include_non_chat=False, include_unknown=False, thinking_budget_on=2048)
    a = config["model_list"][0]
    assert a["litellm_params"]["api_base"] == "http://gpu1:8000/v1"
    assert a["litellm_params"]["model"] == "openai/qwen"


def test_generic_nonboolean_toggle():
    src = {"model": "example", "reasoning_effort": "medium"}
    out = rewrite_request(src, {"example": {"parameter": "chat_template_kwargs.thinking_mode", "on": "enabled", "off": "disabled"}}, "completion")
    assert out["extra_body"]["chat_template_kwargs"]["thinking_mode"] == "enabled"


def test_reasoning_model_info_and_plain_api_key():
    config, mapping, _ = build_fixture()
    info = {x["model_name"]: x for x in config["model_list"]}
    for name in (QWEN, GEMMA, SEED):
        assert info[name]["model_info"]["supports_reasoning"] is True
        assert info[name]["model_info"]["reasoning_effort_levels"] == ["medium"]
        assert info[name]["litellm_params"]["api_key"] == "my-direct-api-key"
    assert info[DEEPSEEK]["model_info"]["supports_reasoning"] is True
    assert info[DEEPSEEK]["model_info"]["reasoning_effort_levels"] == ["low", "high", "max"]


def test_nonchat_model_info_reasoning_false():
    config, _, _ = build(fixture(), {}, api_base="http://localhost:8000/v1", api_key="plain-secret", provider="openai", include_non_chat=True, include_unknown=False, thinking_budget_on=2048)
    entry = next(x for x in config["model_list"] if x["model_name"] == EMBEDDING)
    assert entry["model_info"]["supports_reasoning"] is False
    assert entry["model_info"]["reasoning_effort_levels"] == []


def test_env_reference_api_key_is_rejected():
    with pytest.raises(ValueError, match="literal"):
        build(fixture(), {}, api_base="http://localhost:8000/v1", api_key="os.environ/VLLM_API_KEY", provider="openai", include_non_chat=False, include_unknown=False, thinking_budget_on=2048)
    with pytest.raises(ValueError, match="literal"):
        build(fixture(), {QWEN: {"api_key": "os.environ/SECRET"}}, api_base="http://localhost:8000/v1", api_key="direct", provider="openai", include_non_chat=False, include_unknown=False, thinking_budget_on=2048)
