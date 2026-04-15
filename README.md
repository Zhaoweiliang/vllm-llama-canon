# vllm-llama-canon

An [out-of-tree vLLM plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html)
that adds support for the `LlamaCanonForCausalLM` architecture — the
"canon layer" variant of Llama introduced in Zeyuan Allen-Zhu's
[PhysicsLM4 / Canon Layers](https://ssrn.com/abstract=5240330) work.

A canon layer is a depthwise causal short convolution (kernel=4 by
default) inserted at up to four positions in each decoder block:

- **canonA** — on the residual stream after `input_layernorm`, before attention
- **canonB** — on the fused `qkv` stream before RoPE
- **canonC** — on the residual stream after `post_attention_layernorm`, before MLP
- **canonD** — on the fused `gate_up` stream before `silu * mul`

## Install

```bash
pip install vllm-llama-canon
```

After install, vLLM auto-discovers the plugin via its
`vllm.general_plugins` entry point.

## Use

Pass `trust_remote_code=True` so HuggingFace autoloads the custom
`LlamaCanonConfig` from your model directory:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/your/canon-model",
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="bfloat16",
    enforce_eager=True,
)
print(llm.generate(["hello"], SamplingParams(temperature=0, max_tokens=32))[0].outputs[0].text)
```

Or start a vLLM server:

```bash
vllm serve /path/to/your/canon-model \
  --trust-remote-code --tensor-parallel-size 1 --dtype bfloat16 \
  --enforce-eager --port 8000 --served-model-name canon
```

## What the plugin does

- Registers `LlamaCanonForCausalLM` in `ModelRegistry` via the
  `vllm.general_plugins` entry point — no edits to the vLLM source tree.
- Rebuilds the Llama block with vLLM primitives (`QKVParallelLinear`,
  `MergedColumnParallelLinear`, paged attention, partial RoPE via
  `partial_rotary_factor`) and inserts the four canon convolutions at
  the HF reference positions.
- Each canon conv is a `MambaBase` with `mamba_type="short_conv"` so
  vLLM's V1 engine allocates a per-request `(kernel-1, dim)` rolling
  state alongside the KV cache. The model declares `HasInnerState` and
  `IsHybrid` so the engine plumbs that state correctly.
- The conv forward is written in pure PyTorch (`F.conv1d` for prefill,
  shift-append + dot for decode). The triton kernel
  (`causal_conv1d_fn` / `causal_conv1d_update`) produced state-update
  results that diverged from the reference in this setting; the canon
  width is tiny so the pure-torch path is fine.
- The HF checkpoint loads via vLLM's standard stacked-weight mapping:
  `q_proj/k_proj/v_proj → qkv_proj`, `gate_proj/up_proj → gate_up_proj`,
  canon weights by name. `lm_head` is tied to `embed_tokens` when
  `tie_word_embeddings=True`.

## Limitations

- **`tensor_parallel_size=1` only.** Canon B and canon D operate on
  fused QKV / gate_up streams; per-shard weight layouts under TP>1 need
  separate work.
- **`rope_version='huggingface'` only.** Lingua-style interleaved RoPE
  is not supported.
- **`enforce_eager=True` recommended.** The model class is not
  decorated with `@support_torch_compile`; adding it would require an
  explicit `dynamic_arg_dims`.

## Compatibility

- vLLM `>=0.15,<0.17` (tested on 0.15.1)
- `transformers >= 4.57`
- PyTorch `>= 2.5`
- Python `>= 3.10`

## Parity

Verified against HuggingFace `.generate()` on the
`qwen1.5-0.5b-newtok-canon` PhysicsLM4 checkpoint: 16/16 greedy tokens
match for both a 1-token prompt (exercises the decode-path conv state
update) and a 12-token prompt (exercises the prefill conv).

## License

Apache 2.0. See [LICENSE](LICENSE).
