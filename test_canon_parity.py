"""Correctness check: compare vLLM canon model output against HuggingFace reference.

Computes the next-token logits for the same prompt through both paths and
reports the max absolute difference and top-k argmax agreement.

Run:
    conda activate vllm
    CUDA_VISIBLE_DEVICES=4 python test_canon_parity.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import importlib.util
import sys

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


def _preload_custom_modules(model_dir: str) -> None:
    """HF's dynamic_module loader rejects relative imports like
    `from canon_helper import ...`. Preload the sibling files as top-level
    modules so the import check passes."""
    for fname in ("canon_helper", "configuration_llama_canon"):
        if fname in sys.modules:
            continue
        path = os.path.join(model_dir, f"{fname}.py")
        spec = importlib.util.spec_from_file_location(fname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fname] = mod
        spec.loader.exec_module(mod)

MODEL = "/local/data/weiliang/sycl/train/PhysicsLM4/qwen1.5-0.5b-newtok-canon"


def hf_greedy(prompt_ids: list[int], max_new_tokens: int = 16) -> list[int]:
    _preload_custom_modules(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float32
    ).to("cuda").eval()
    input_ids = torch.tensor([prompt_ids], device="cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            num_beams=1,
            use_cache=True,
            pad_token_id=tok.pad_token_id or 0,
        )
    generated = out[0].tolist()[len(prompt_ids):]
    del model
    torch.cuda.empty_cache()
    return generated


def vllm_greedy(prompt_ids: list[int], max_new_tokens: int = 16) -> list[int]:
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="float32",
        gpu_memory_utilization=0.3,
        max_model_len=2048,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)
    return list(out[0].outputs[0].token_ids)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # Short prompt (vLLM classifies as decode-path on first step)
    short_ids = tok.encode("The quick brown fox", add_special_tokens=True)
    # Multi-token prompt (hits prefill path)
    long_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    print("short prompt_ids:", short_ids)
    print("long prompt_ids: ", long_ids)

    for label, ids in (("short", short_ids), ("long", long_ids)):
        print(f"\n== {label} prompt ==")
        hf_out = hf_greedy(ids)
        print("HF:   ", hf_out)
        vl_out = vllm_greedy(ids)
        print("vLLM: ", vl_out)
        match = sum(a == b for a, b in zip(hf_out, vl_out))
        print(f"prefix-match: {match}/{len(hf_out)} tokens")


if __name__ == "__main__":
    main()
