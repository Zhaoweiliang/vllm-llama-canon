"""vLLM model for `LlamaCanonForCausalLM` (canon layers from PhysicsLM4).

Matches the HuggingFace reference in
`canon_helper.py` + `modeling_llama_canon.py` at
`train/PhysicsLM4/qwen1.5-0.5b-newtok-canon/`.

Limitations:
- Only tensor_parallel_size=1 is supported. Canon B (fused QKV) and canon D
  (fused gate/up) would require matching per-shard weight layouts to vLLM's
  QKVParallelLinear / MergedColumnParallelLinear under TP>1.
- `rope_version='lingua'` is not supported.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .canon_layer import CanonConv


def _build_canon(
    *,
    name: str,
    dim: int,
    config,
    vllm_config: VllmConfig,
    prefix: str,
) -> CanonConv:
    return CanonConv(
        dim=dim,
        kernel_size=config.canon_kernel,
        bias=config.canon_bias,
        activation="silu" if config.canon_activation else None,
        residual=config.canon_residual,
        model_config=vllm_config.model_config,
        cache_config=vllm_config.cache_config,
        prefix=f"{prefix}.{name}",
    )


class LlamaCanonMLP(nn.Module):
    def __init__(
        self,
        *,
        config,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=config.mlp_bias,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=config.mlp_bias,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation {config.hidden_act!r}")
        self.act_fn = SiluAndMul()
        if "D" in config.canon_set:
            self.canonD = _build_canon(
                name="canonD",
                dim=2 * intermediate_size,
                config=config,
                vllm_config=vllm_config,
                prefix=prefix,
            )
        else:
            self.canonD = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        if self.canonD is not None:
            gate_up = self.canonD(gate_up)
        y = self.act_fn(gate_up)
        out, _ = self.down_proj(y)
        return out


class LlamaCanonAttention(nn.Module):
    def __init__(
        self,
        *,
        config,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise NotImplementedError(
                "LlamaCanon plugin currently supports tensor_parallel_size=1 only "
                "(canon B/D weight sharding is not implemented for TP>1)."
            )

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.attention_bias,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,  # HF model: o_proj has no bias
            prefix=f"{prefix}.o_proj",
        )

        # Optional Q/K norm (qk_norm=False in the shipped config but support it)
        if getattr(config, "qk_norm", False):
            self.q_norm = RMSNorm(
                self.total_num_heads * self.head_dim, eps=config.rms_norm_eps
            )
            self.k_norm = RMSNorm(
                self.total_num_kv_heads * self.head_dim, eps=config.rms_norm_eps
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Optional canonB over the fused qkv stream
        if "B" in config.canon_set:
            total_dim = (
                self.total_num_heads * self.head_dim
                + 2 * self.total_num_kv_heads * self.head_dim
            )
            self.canonB = _build_canon(
                name="canonB",
                dim=total_dim,
                config=config,
                vllm_config=vllm_config,
                prefix=prefix,
            )
        else:
            self.canonB = None

        # RoPE: partial rotary via rope_dim -> partial_rotary_factor
        partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
        if partial_rotary_factor is None:
            rope_dim = getattr(config, "rope_dim", None)
            partial_rotary_factor = (
                1.0 if rope_dim is None else rope_dim / self.head_dim
            )
        rope_version = getattr(config, "rope_version", "huggingface")
        if rope_version != "huggingface":
            raise NotImplementedError(
                f"rope_version={rope_version!r} not supported in vLLM plugin"
            )
        rope_parameters = {
            "rope_theta": float(config.rope_theta),
            "rope_type": "default",
            "partial_rotary_factor": float(partial_rotary_factor),
        }
        if config.rope_scaling:
            rope_parameters.update(config.rope_scaling)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        # Q/K norm is applied on q and k (pre-canon in the HF impl).
        if self.q_norm is not None or self.k_norm is not None:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)
            qkv = torch.cat([q, k, v], dim=-1)

        if self.canonB is not None:
            qkv = self.canonB(qkv)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaCanonDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaCanonAttention(
            config=config,
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaCanonMLP(
            config=config,
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.canonA = (
            _build_canon(
                name="canonA",
                dim=config.hidden_size,
                config=config,
                vllm_config=vllm_config,
                prefix=prefix,
            )
            if "A" in config.canon_set
            else None
        )
        self.canonC = (
            _build_canon(
                name="canonC",
                dim=config.hidden_size,
                config=config,
                vllm_config=vllm_config,
                prefix=prefix,
            )
            if "C" in config.canon_set
            else None
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.canonA is not None:
            hidden_states = self.canonA(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.canonC is not None:
            hidden_states = self.canonC(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaCanonModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaCanonDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ---- weight-loading -----------------------------------------------------
# HF checkpoint names we need to remap:
#
#   model.embed_tokens.weight          -> same
#   model.layers.{i}.self_attn.q_proj.{weight,bias}   -> qkv_proj (shard "q")
#   model.layers.{i}.self_attn.k_proj.{weight,bias}   -> qkv_proj (shard "k")
#   model.layers.{i}.self_attn.v_proj.{weight,bias}   -> qkv_proj (shard "v")
#   model.layers.{i}.self_attn.o_proj.weight          -> o_proj
#   model.layers.{i}.self_attn.canonB.weight          -> self_attn.canonB.weight
#   model.layers.{i}.mlp.gate_proj.weight             -> gate_up_proj (shard 0)
#   model.layers.{i}.mlp.up_proj.weight               -> gate_up_proj (shard 1)
#   model.layers.{i}.mlp.down_proj.weight             -> down_proj
#   model.layers.{i}.mlp.canonD.weight                -> mlp.canonD.weight
#   model.layers.{i}.canonA.weight                    -> same (under DecoderLayer)
#   model.layers.{i}.canonC.weight                    -> same
#   model.layers.{i}.input_layernorm.weight           -> same
#   model.layers.{i}.post_attention_layernorm.weight  -> same
#   model.norm.weight                                  -> same
#   lm_head.weight (absent when tie_word_embeddings)  -> tied


class LlamaCanonForCausalLM(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        """Return the largest canon-state shape for page-size calculation."""
        hf = vllm_config.model_config.hf_config
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        dims = []
        if "A" in hf.canon_set:
            dims.append(hf.hidden_size)
        if "B" in hf.canon_set:
            dims.append(
                hf.num_attention_heads * head_dim
                + 2 * hf.num_key_value_heads * head_dim
            )
        if "C" in hf.canon_set:
            dims.append(hf.hidden_size)
        if "D" in hf.canon_set:
            dims.append(2 * hf.intermediate_size)
        if not dims:
            dims.append(hf.hidden_size)
        max_dim = max(dims)
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=max_dim,
            conv_kernel=hf.canon_kernel,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config

        self.model = LlamaCanonModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("lm_head.") and self.config.tie_word_embeddings:
                continue
            matched = False
            for dst, src, shard in stacked_params_mapping:
                if src not in name:
                    continue
                new_name = name.replace(src, dst)
                if new_name.endswith(".bias") and new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                param.weight_loader(param, weight, shard)
                loaded.add(new_name)
                matched = True
                break
            if matched:
                continue
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                # Unknown tensor — skip rather than crash so custom buffers
                # from the HF module don't block loading.
                continue
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)
            loaded.add(name)
        return loaded
