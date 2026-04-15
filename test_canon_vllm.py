"""Smoke test: load the canon model with vLLM and generate a few tokens.

Run:
    conda activate vllm
    CUDA_VISIBLE_DEVICES=4 python test_canon_vllm.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "/local/data/weiliang/sycl/train/PhysicsLM4/qwen1.5-0.5b-newtok-canon"


def main() -> None:
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="float32",
        gpu_memory_utilization=0.3,
        max_model_len=2048,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    outs = llm.generate(["Hello, my name is"], sp)
    for o in outs:
        print("PROMPT:", repr(o.prompt))
        print("OUTPUT:", repr(o.outputs[0].text))


if __name__ == "__main__":
    main()
