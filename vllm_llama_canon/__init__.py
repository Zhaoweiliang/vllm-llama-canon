def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "LlamaCanonForCausalLM",
        "vllm_llama_canon.modeling_llama_canon:LlamaCanonForCausalLM",
    )
