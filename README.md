# LiteLLM + vLLM model capabilities generator

Generates `config.yaml` for LiteLLM Proxy, `thinking_rules.json`, `thinking_hook.py`, and `thinking_transform.py` from the previously created `vllm_model_capabilities.json`. Requires Python 3.10+ and PyYAML. LiteLLM Proxy requires `litellm[proxy]` in the runtime environment.

## Generate

```bash
pip install pyyaml pytest
python generate_litellm_config.py \
  --input vllm_model_capabilities.json \
  --output-dir generated \
  --api-base os.environ/VLLM_API_BASE \
  --api-key my-vllm-api-key \
  --thinking-budget-on 2048
```

`litellm_params.api_key` is **always written as a literal string**: use `--api-key` for the default or a literal `api_key` in the endpoint map for per-model overrides. `os.environ/...` references are rejected. Avoid checking generated `config.yaml` into version control, and restrict file permissions.

Each generated `model_info` includes `supports_reasoning` and `reasoning_effort_levels`:
- Toggle-only: `supports_reasoning: true`, `reasoning_effort_levels: [medium]` (`null`/omitted is OFF, not a level).
- Native reasoning-effort: `supports_reasoning: true` and source-declared supported levels.
- Unsupported: `supports_reasoning: false` and empty levels.
- Thinking-supported but no verified effort levels: `supports_reasoning: true` and empty levels; do not treat empty levels as an authoritative list of supported values.

Note: LiteLLM may round non-matching effort levels using `reasoning_effort_levels`. The hook itself rejects values other than `medium`/null for toggle-only models if it receives them unchanged.

The CLI **does not hardcode `reasoning_effort`** into any deployment. Only models with `thinking.supported=true`, a `thinking.toggle`, and *no native* `reasoning_effort.parameter` or nonempty `reasoning_effort.options` get a toggle rule.

For toggle-only models, the app's `reasoning_effort` (either top-level or inside `extra_body`) is converted by the proxy:

| App value | vLLM effect |
| --- | --- |
| omitted or `null` | OFF |
| `"medium"` | ON |
| anything else, including `"middle"` | 400 Bad Request |

For native-effort models, no transform occurs. The model receives the original `reasoning_effort` parameter. `middle` is not rewritten or accepted by the toggle conversion.

The hook is registered in YAML as **the instance** `thinking_hook.thinking_hook` and is executed for Chat Completions only (`call_type="completion"`). Other API routes are untouched.

## Run

```bash
export VLLM_API_BASE=http://localhost:8000/v1
export LITELLM_MASTER_KEY=sk-example-secret
cd generated
litellm --config config.yaml --port 4000
```

The LiteLLM process must see the directory containing the two hook Python files in `PYTHONPATH` (starting it from `generated` as shown accomplishes this). If needed set `THINKING_RULES_PATH=/absolute/path/to/generated/thinking_rules.json`.

App example:

```json
{
  "model": "Qwen/Qwen3.6-35B-A3B",
  "messages": [{"role": "user", "content": "Explain this."}],
  "reasoning_effort": "medium"
}
```

vLLM receives `chat_template_kwargs.enable_thinking=true` and no `reasoning_effort` for this toggle-only model. When the application omits `reasoning_effort`, vLLM receives `enable_thinking=false`.

## Multi-vLLM deployment mapping

vLLM normally serves one model per server. For different URLs, use an endpoint map:

```json
{
  "Qwen/Qwen3.6-35B-A3B": {
    "api_base": "http://gpu1:8000/v1",
    "served_model_name": "qwen-3.6",
    "api_key": "direct-key-for-gpu1"
  },
  "ByteDance-Seed/Seed-OSS-36B-Instruct": "http://gpu2:8000/v1"
}
```

Then append `--endpoint-map endpoints.json`. Models not present use the default `--api-base`. The `served_model_name` must match the name returned by the upstream vLLM's `/v1/models` endpoint.

The provider defaults to `openai/<served_name>` for OpenAI-compatible API pass-through. Pass `--provider hosted_vllm` to use LiteLLM's hosted-vLLM integration instead; verify parameter forwarding on your LiteLLM version.

## Non-chat models

Default config contains only `chat_supported=true` models. `--include-non-chat` adds embeddings, image generation, STT and TTS where common OpenAI-compatible endpoint mappings are known, with `model_info.enforce_supported_endpoints: true` to prevent chat requests from being routed to them. OCR/classifier and models with unknown chat support are skipped. You may add `--include-unknown` to opt in unknown models **only after verifying support**. Endpoint annotations do not guarantee LiteLLM or a specific vLLM server actually implements a given model's task.

## Design and caveats

- Model metadata (`chat_supported`, `thinking.supported` and effort options) must be verified against the exact model revision, vLLM version and chat template before production use; no live compatibility probe is performed.
- An `on` value equal to `"positive_integer_budget"` is replaced by `--thinking-budget-on` (default 2048). A model can otherwise supply boolean, string or numeric `toggle.on/off` values directly. All mapped toggle parameters are written through `extra_body`; e.g. `chat_template_kwargs.enable_thinking`, `chat_template_kwargs.thinking_budget` and `chat_template_kwargs.thinking_mode`.
- Native effort detection is conservative. A model with `reasoning_effort.parameter` set is left untouched, even when its `options` are unknown.
- The hook is only for LiteLLM Proxy Chat Completions, not the direct LiteLLM SDK, Responses API, or other endpoint types.
- When the client uses a Python OpenAI SDK and passes `reasoning_effort=None`, the field may be omitted in JSON; the hook deliberately treats omission as OFF.
- The public `model_name` is the metadata key. Do not rename it unless you also update the lookup keys in `thinking_rules.json`.

## Tests

```bash
pytest -q tests/
```
