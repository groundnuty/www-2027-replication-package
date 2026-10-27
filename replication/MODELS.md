# Models and their settings

This file lists the models of the reported runs, their checkpoints, the model-side serving flags, the configurations
used for each graph and run group, the sampling the runs used, and the harness settings per model call. The code that applies the harness settings
is in `harness/agent/providers.py` (routes) and `scripts/run_pilot.py` (the model table, `MODELS`); each run's manifest
records the values it ran with.

The open-weight models were served with vLLM (OpenAI-compatible server). Checkpoints are the Hugging Face repositories below
at the stated commit, loaded from local copies of those revisions (offline mode). No `--chat-template` override and no
`--override-generation-config` was passed to the servers we configured: each model used the chat template shipped in its checkpoint revision
(`chat_template.jinja`, SHA-256 prefix given), except DeepSeek, which uses its own encoder (`--tokenizer-mode
deepseek_v4`). Scheduling, memory-budget, loading-only and network settings are not listed.
Hardware-specific serving flags (tensor parallelism, memory fraction, ports) are omitted.

## Model-side flags

| Model | Checkpoint (repo @ commit) | vLLM | Weights | max-model-len | Tool-call parser | Reasoning parser | Chat template | Other flags |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8 | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 0.29.0 | bfloat16, no quantization | 262,144 | `qwen3_coder` | `qwen3` | checkpoint's own (sha256 `c3cf9e34abf4f9e3...`) | automatic prefix caching; automatic tool choice; attention backend FLASH_ATTN (vLLM's choice) |
| Gemma-4 | `RedHatAI/gemma-4-31B-it-FP8-block` @ `d7242548c457ab4b45bd3adb8937f2659af4739d`, served under the names `google/gemma-4-31B` and the repo id | 0.32.0 | FP8 block-wise weights (compressed-tensors, as shipped); model dtype bfloat16 | 262,144 | `gemma4` | `gemma4` | checkpoint's own (sha256 `6a1015c47ccfcfa6...`) | prefix caching; automatic tool choice; FP8 path and attention kernel per configuration, below |
| GLM-5.3-Flash | `zai-org/GLM-5.3-Flash` @ `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` | 0.32.0 | FP8 weights (as shipped) | 1,048,576 (checkpoint default) | `glm47` | `glm45` | checkpoint's own (sha256 `2614a3c896d53e26...`) | `--trust-remote-code`; `--no-enable-flashinfer-autotune`; prefix caching; attention per configuration, below |
| DeepSeek-V4-Flash-0731 | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | 0.26.0 | FP8 weights (as shipped; bfloat16 activations); KV cache FP8 (`--kv-cache-dtype fp8`) | 700,000 (the checkpoint's position limit is 1,048,576 with YaRN factor 16 from its config; served at 700,000) | `deepseek_v4` | `deepseek_v4` | none: the model's own encoder (`--tokenizer-mode deepseek_v4`; tokenizer_config sha256 `6ac8c8dc065ed118...`) | `--trust-remote-code`; prefix caching and chunked prefill |
| Qwen3-Coder-30B-A3B-Instruct | `Qwen/Qwen3-Coder-30B-A3B-Instruct` @ `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` | 0.15.0 | bfloat16 (`--dtype auto`), no quantization | 249,600 (checkpoint limit 262,144; served lower) | `qwen3_coder` | none | checkpoint's own (sha256 `5a38bfa058332662...`) | automatic tool choice; default attention backend; no prefix-caching flag set |

Reasoning content, where a model emits it, is parsed by the parser named above. No `chat_template_kwargs` or other
extra request body was sent for any model (the manifests record `extra_body` as absent).

## Configurations by graph and run group

Where a model has one configuration for all its reported runs, it has one row. Where it had two, each is listed with the
graphs and run groups it served. "Turns of both configurations; the data record each turn's configuration" means that a
run group holds turns of both.

| Model | Configuration and the graphs and run groups it served |
|---|---|
| Qwen3.8 | **Configuration A (flags as in the table above):** Rhea, opaque Rhea, the private graph and the SIB runs (the Rhea originals and the Rhea clamp re-runs; the UniProt originals and the UniProt partitions). **Configuration B (served through a hosted endpoint that adds a short fixed segment to every prompt):** the UniProt runs (natural, baseline and retry cells) and corporate. Through that endpoint the chat template differs from the checkpoint's (text not recorded); engine release 0.29.0 and context window 262,144 as above; other settings not exposed by the endpoint. |
| Gemma-4 | **Configuration A:** weight-only FP8 (bfloat16 activations, the checkpoint's float32 block scales) with Triton attention. **Configuration B:** FP8 activations per 128-element group with power-of-two (E8M0) block scales, FlashAttention 4 attention. Corporate: A. UniProt SPARQL-LLM baseline cells: one run directory per configuration. UniProt (natural, baseline and retry cells), Rhea and the private graph: turns of both configurations; the data record each turn's configuration. All other flags are identical. |
| GLM-5.3-Flash | **Configuration A:** the 11 sparse-attention layers attend to all earlier tokens (a local patch); **configuration B:** top-2048 sparse attention (the model's indexer). The two are identical up to 2,048 tokens of context. UniProt initial outputs and the private graph: A. Rhea, opaque Rhea, corporate, the UniProt baseline cells: B. UniProt retry cells, and the baseline re-runs and opaque Rhea partitions that continued earlier outputs: turns of both configurations; the data record each turn's configuration. The FP8 linear path may differ between A and B in the way it does for Gemma-4; this was not verified from engine logs. All other flags are identical. |
| DeepSeek-V4-Flash-0731 | **Configuration A (flags as in the table above):** Rhea, opaque Rhea, corporate and the private graph. **Configuration B (served through a hosted endpoint that adds a short fixed segment to every prompt):** the UniProt initial outputs and baseline cells; engine release 0.26.0, FP8 KV cache, context window 700,000 and the parsers above as declared by its operator; other settings not exposed by the endpoint. UniProt retry cells: turns of both configurations; the data record each turn's configuration. |
| Qwen3-Coder-30B-A3B-Instruct | **Configuration A (flags as in the table above):** Rhea, opaque Rhea and the private graph. **Configuration B (served through a hosted endpoint that adds a short fixed segment to every prompt):** the UniProt runs (natural, baseline and retry cells) and corporate; context window 249,600; engine release not recorded; other settings not exposed by the endpoint. |
| Claude Sonnet 5.5, Claude Haiku 5.5 | One configuration (below). |

## Sampling

The harness sends no temperature, top_p, top_k or penalties (`harness/agent/providers.py`, `OpenAICompatRoute`: a
parameter is sent only when given, and none is given). A server we configured therefore applies the defaults of the
checkpoint's `generation_config.json` (vLLM `--generation-config auto`, its default); the hosted endpoint's sampling
defaults are not exposed by the endpoint.

| Model | temperature | top_p | top_k | repetition penalty |
|---|---|---|---|---|
| Qwen3.8 | 1.0 | 0.95 | 20 | none |
| Gemma-4 | 1.0 | 0.95 | 64 | none |
| GLM-5.3-Flash | 1.0 | 0.95 | server default (off) | none |
| DeepSeek-V4-Flash-0731 | 1.0 | 1.0 | server default (off) | none |
| Qwen3-Coder-30B-A3B-Instruct | 0.7 | 0.8 | 20 | 1.05 |

Seed policy: no sampling seed is sent in the main runs. The retry arms of the checker loop send a per-arm sampling seed
on the models that accept one (the vLLM models; the Claude API takes none): the first 31 bits of
sha256("<order seed>|<parent run id>|<arm>") (`harness/checkloop/online.py`, `sampling_seed`). The vLLM engine seed is
left at its default.

## Harness settings per model call

The values pinned in the model table of the framework (`scripts/run_pilot.py`, `MODELS`), checked against the run
manifests (settings fields only). "Turn cap" is model calls per attempt; every attempt also has a wall limit of 7,200 s.

| Model | Output cap per call (tokens) | Turn cap: UniProt, Rhea, corporate, private graph | Turn cap: opaque Rhea | Request time-out | Transport |
|---|---|---|---|---|---|
| Qwen3.8 | 32,768 (81,920 in the SIB runs) | 240 | 240 | 600 s in the UniProt and corporate runs; 1,800 s (runner setting) in the Rhea, opaque Rhea and private-graph runs; 3,577 s in the SIB runs | streamed, with the stream guards |
| Gemma-4 | 16,000 | 120 | 240 (runner setting) | 600 s | streamed, with the stream guards |
| GLM-5.3-Flash | 32,768 | 240 | 240 | 600 s | streamed, with the stream guards |
| DeepSeek-V4-Flash-0731 | 16,000 | 120 | 240 (runner setting) | 600 s | not streamed |
| Qwen3-Coder-30B-A3B-Instruct | 16,000 | 120 | 240 (runner setting) | 600 s | streamed, with the stream guards |
| Claude Sonnet 5.5 | 16,000 (includes thinking tokens) | 120 | 120 | 600 s | not streamed |
| Claude Haiku 5.5 | 16,000 (includes thinking tokens) | 120 | 120 | 600 s | not streamed |

The manifests read confirm: 120 turns and 600 s for every Claude run (opaque Rhea included); 240 turns for every Qwen3.8
and GLM run (opaque Rhea included); 1,800 s for the private-graph Qwen3.8 runs started under code tag `test-e180`; and
for every Qwen3.8 SIB run (the Rhea originals, the Rhea clamp re-runs, the UniProt originals and the five UniProt
partitions) an output cap of 81,920 tokens, 240 turns and a request time-out of 3,577 s.
The 240-turn cap of Gemma, DeepSeek and Coder in opaque Rhea runs is the per-model runner setting of that tag.

Retries are made by the harness, not the SDK: a transient error (rate limit, 5xx, time-out, connection error) is
retried up to 4 times with exponential backoff (2 s base, 60 s maximum); after the last retry the attempt is recorded as
an infrastructure failure and is not scored. The stream guards (`guard_call_identity`, `guard_token_gap` in
`harness/agent/providers.py`) reject a streamed reply with a tool call that has no id or no function name, or a chunk
whose completion-token counter rose by at least 30 while it delivered fewer than 0.5 characters per token; a rejected
stream is retried like any transient error. The model table also holds exploratory-candidate models (Llama 3.3 70B,
Bielik 11B, Qwen3.6 35B-A3B) that are not part of the reported panel.

### Raised output caps (new runs only)

The code tags listed in the README (`Versions`) raised the per-call output cap for the cap-sensitivity runs, together
with the request time-out of a whole turn at that cap:

| Tag (README) | Models | Output cap per call | Request time-out |
|---|---|---|---|
| A (`claude32k-20261024-1735e82a`) | Claude Sonnet 5.5 (the tag raises Claude Haiku 5.5's cap too) | 32,000 | 600 s |
| A2 (`claudemax-20261024-8bc0b35a`) | Claude Sonnet 5.5 (the tag raises Claude Haiku 5.5's cap too) | 128,000 | 1,400 s |
| B (`cap81920-qwen-deepseek-20261024-34817307`) | Qwen3.8 (SIB runs and cap-sensitivity runs), DeepSeek-V4-Flash-0731 | 81,920 | 3,577 s for the Qwen3.8 SIB runs (recorded in their manifests); 3,600 s (per-model runner setting) for the other tag-B runs |
| C (`cap81920-glm-coder-20261024-12c0b79d`) | GLM-5.3-Flash, Qwen3-Coder-30B-A3B-Instruct | 81,920 | 5,400 s (GLM), 3,600 s (Coder) |

DeepSeek's cap-sensitivity runs also used the 240-turn cap. Sonnet's SIB runs used tags A and A2.

## Claude models

| | Claude Sonnet 5.5 | Claude Haiku 5.5 |
|---|---|---|
| API model id (as recorded in the manifests) | `claude-sonnet-5-5` | `claude-haiku-5-5` |
| Output cap per call | 16,000 tokens (thinking included) | 16,000 tokens (thinking included) |
| Thinking | adaptive (`{"type": "adaptive"}`), sent explicitly | adaptive, sent explicitly |
| Effort | `high`, sent explicitly | `high`, sent explicitly |
| Temperature, top_p, top_k | not sent (API defaults) | not sent (API defaults) |
| Seed | the Messages API takes none | none |
| Prompt cache | 1-hour cache on the static prefix | 1-hour cache on the static prefix |

The same settings apply to Claude Opus 5.5 (`claude-opus-5-5`), which ran only in the exploratory self-check. The API
exposes no revision identifier beyond the model id.
