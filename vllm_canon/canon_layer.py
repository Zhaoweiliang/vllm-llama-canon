"""Canon convolution layer for vLLM.

A canon layer (from PhysicsLM4, Zeyuan Allen-Zhu) is a depthwise causal short
convolution with optional activation and a residual add. This module plugs into
vLLM's V1 engine as a MambaBase so the engine allocates per-request state for
us (one (kernel-1)-wide rolling buffer per channel).

Implementation note: we use a plain PyTorch implementation for the
prefill+decode paths instead of vLLM's `causal_conv1d_fn` / `causal_conv1d_update`
kernels. The vLLM kernels were giving state-update results that diverged from
the reference implementation in this setting, and the pure-torch version matches
HuggingFace parity exactly. The layers are small (state width = 3), so the
performance cost is negligible.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata


@CustomOp.register("canon_conv")
class CanonConv(MambaBase, CustomOp):
    """Depthwise causal short convolution with per-request state, used by the
    canon layers from Zeyuan Allen-Zhu's PhysicsLM4 work.

    Behavior mirrors `canon_helper.ShortConvolution`:
      input:  (num_tokens, dim)       # vLLM flattens across requests
      weight: (dim, 1, kernel)        # depthwise, groups=dim
      output: (num_tokens, dim)
      output = conv(input) [+ input]  # residual optional
    """

    def __init__(
        self,
        *,
        dim: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = None,
        residual: bool = True,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert activation in (None, "silu", "swish"), activation
        assert kernel_size >= 2
        self.dim = dim
        self.kernel_size = kernel_size
        self.activation = "silu" if activation == "swish" else activation
        self.residual = residual
        self.prefix = prefix

        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]),)

        self.model_config = model_config
        self.cache_config = cache_config

    def get_state_shape(self):
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=self.dim,
            conv_kernel=self.kernel_size,
        )

    def get_state_dtype(self):
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    @property
    def mamba_type(self) -> str:
        return "short_conv"

    def forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.canon_conv(hidden_states, output, self.prefix)
        return output

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Mutate `output` in-place: output[:N] = conv(hidden_states[:N]) [+ residual].
        `hidden_states` is flat (num_tokens, dim); state is sliced per request.
        """
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run / no routing metadata: pass-through identity if
            # residual, else zeros.
            if self.residual:
                output.copy_(hidden_states)
            else:
                output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        meta: ShortConvAttentionMetadata = attn_metadata[self.prefix]
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        # conv_state storage is (num_slots, kernel-1, dim); view as (num_slots, dim, kernel-1)
        conv_state = self_kv_cache[0].transpose(-1, -2)

        state_indices = meta.state_indices_tensor
        query_start_loc_p = meta.query_start_loc_p
        num_prefills = meta.num_prefills
        num_decodes = meta.num_decode_tokens
        num_prefill_tokens = meta.num_prefill_tokens
        num_actual_tokens = num_decodes + num_prefill_tokens

        W = self.kernel_size
        D = self.dim
        weight = self.weight.view(D, W)

        si_d, si_p = torch.split(
            state_indices, [num_prefills and num_decodes or num_decodes, num_prefills], dim=0
        ) if False else (state_indices[:num_decodes], state_indices[num_decodes:])

        y = torch.empty_like(hidden_states[:num_actual_tokens])

        # V1 layout: decode tokens come first, then prefill tokens
        if num_decodes > 0:
            self._run_decode(
                hidden_states[:num_decodes],
                conv_state,
                si_d,
                weight,
                y[:num_decodes],
            )
        if num_prefills > 0:
            assert query_start_loc_p is not None
            self._run_prefill(
                hidden_states[num_decodes:num_actual_tokens],
                conv_state,
                si_p,
                weight,
                query_start_loc_p,
                y[num_decodes:],
            )

        if self.residual:
            output[:num_actual_tokens].copy_(hidden_states[:num_actual_tokens] + y)
        else:
            output[:num_actual_tokens].copy_(y)

    def _run_decode(
        self,
        x_d: torch.Tensor,          # (nD, D)
        conv_state: torch.Tensor,    # (num_slots, D, W-1)
        si_d: torch.Tensor,          # (nD,)
        weight: torch.Tensor,        # (D, W)
        out: torch.Tensor,           # (nD, D), written to
    ) -> None:
        W = self.kernel_size
        # Gather past state per request: (nD, D, W-1)
        si_d_long = si_d.to(torch.long)
        s = conv_state.index_select(0, si_d_long)
        # Concatenate past state + current token along width -> (nD, D, W)
        window = torch.cat([s, x_d.unsqueeze(-1)], dim=-1)
        # Depthwise conv output = sum(window * weight) over width
        y = (window * weight.unsqueeze(0)).sum(dim=-1)  # (nD, D)
        if self.bias is not None:
            y = y + self.bias
        if self.activation == "silu":
            y = F.silu(y)
        out.copy_(y)
        # Write back updated state: drop oldest, append x -> (nD, D, W-1)
        new_state = window[..., 1:]
        conv_state.index_copy_(0, si_d_long, new_state)

    def _run_prefill(
        self,
        x_p: torch.Tensor,           # (total_prefill_tokens, D)
        conv_state: torch.Tensor,    # (num_slots, D, W-1)
        si_p: torch.Tensor,          # (num_prefills,)
        weight: torch.Tensor,        # (D, W)
        query_start_loc: torch.Tensor,  # (num_prefills+1,)
        out: torch.Tensor,           # (total_prefill_tokens, D), written to
    ) -> None:
        W = self.kernel_size
        D = self.dim
        w = weight.unsqueeze(1)  # (D, 1, W)
        qsl = query_start_loc.tolist()
        num_prefills = len(qsl) - 1
        for r in range(num_prefills):
            lo = qsl[r]
            hi = qsl[r + 1]
            T = hi - lo
            if T <= 0:
                continue
            slot = int(si_p[r].item())
            x_r = x_p[lo:hi]  # (T, D)
            x_bdt = x_r.transpose(0, 1).unsqueeze(0)  # (1, D, T)
            xp = F.pad(x_bdt, (W - 1, 0))
            y_bdt = F.conv1d(xp, w, groups=D)  # (1, D, T)
            if self.bias is not None:
                y_bdt = y_bdt + self.bias.view(1, -1, 1)
            if self.activation == "silu":
                y_bdt = F.silu(y_bdt)
            out[lo:hi] = y_bdt.squeeze(0).transpose(0, 1)
            # Update state with the last W-1 tokens (zero-padded if shorter)
            if T >= W - 1:
                tail = x_r[-(W - 1):].transpose(0, 1)  # (D, W-1)
            else:
                pad = torch.zeros(D, W - 1 - T, device=x_r.device, dtype=x_r.dtype)
                tail = torch.cat([pad, x_r.transpose(0, 1)], dim=-1)
            conv_state[slot].copy_(tail)


def _canon_conv(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    ctx: ForwardContext = get_forward_context()
    self: CanonConv = ctx.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states, output=output)


def _canon_conv_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="canon_conv",
    op_func=_canon_conv,
    mutates_args=["output"],
    fake_impl=_canon_conv_fake,
)
