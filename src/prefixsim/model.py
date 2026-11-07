"""KV bytes per token and a prefill-FLOPs cost model, from a Hugging Face config."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

_DTYPE_BYTES = {"float32": 4, "fp32": 4, "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2,
                "fp8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1}
# fields meaning some layers do not keep a full KV cache
_NOT_FULL = ("attention_chunk_size", "full_attention_interval", "attn_layer_period",
             "hybrid_override_pattern", "layers_block_type", "sliding_window_pattern")


@dataclass(frozen=True)
class ModelSpec:
    """What a prefix-cache simulation needs of a model.

    Attributes:
        name: for results, e.g. the Hugging Face repo id.
        kv_bytes_per_token: KV-cache bytes of one token, all layers.
        num_layers: layers that keep a KV cache.
        kv_dtype: the KV-cache element type ``kv_bytes_per_token`` is for.
        active_params: parameters a token passes through, embeddings excluded
            and only the routed experts of an MoE layer; ``None`` if the config
            does not say, and then :meth:`cost_model` needs it set by hand.
        attn_flops_per_token: attention FLOPs a token spends on each earlier
            token, all layers.
    """

    name: str
    kv_bytes_per_token: int
    num_layers: int | None = None
    kv_dtype: str = "bfloat16"
    active_params: int | None = None
    attn_flops_per_token: int | None = None

    def block_bytes(self, block_size_tokens: int) -> int:
        """KV-cache bytes of one block."""
        return self.kv_bytes_per_token * block_size_tokens

    def cost_model(self, block_size_tokens: int, max_blocks_per_request: int = 16384) -> str:
        """PrefixSim's ``linear:A,B``: prefill FLOPs of the block at position p are A + B*p.

        A: 2 FLOPs per active parameter per token, plus attention within the
        block. B: attention over one more block of context.

        Args:
            block_size_tokens: tokens per block, the trace's.
            max_blocks_per_request: the longest request, in blocks: A and B are
                scaled to integers so that its last block's cost fits
                PrefixSim's int32.
        """
        if self.active_params is None or self.attn_flops_per_token is None:
            raise ValueError(f"{self.name}: active_params unknown; pass cost_model= yourself")
        n, attn = block_size_tokens, self.attn_flops_per_token
        a, b = 2 * self.active_params * n + attn * n * (n + 1) / 2, attn * n * n
        unit = min(a, b) / 1000
        while round(a / unit) + round(b / unit) * (max_blocks_per_request - 1) > 2**31 - 1:
            unit *= 2
        a, b = max(1, round(a / unit)), round(b / unit)
        g = math.gcd(a, b)
        return f"linear:{a // g},{b // g}"

    @classmethod
    def from_config(cls, config: dict | str | os.PathLike, *, name: str | None = None,
                    kv_dtype: str = "auto", strict: bool = True) -> ModelSpec:
        """From a Hugging Face ``config.json``: MHA/GQA/MQA or MLA, dense or MoE.

        Args:
            config: the config as a dict, or a path to it.
            name: for results; default the config's ``_name_or_path``.
            kv_dtype: KV-cache element type: ``auto`` (the config's
                ``torch_dtype``), ``bf16``, ``fp16``, ``fp8``, ``fp32``.
            strict: refuse configs where not every layer keeps a full KV cache
                (sliding window, chunked or linear attention). ``False`` counts
                every layer as full attention, an upper bound.
        """
        if not isinstance(config, dict):
            with open(config) as f:
                config = json.load(f)
        c = config.get("text_config") or config  # multimodal configs
        name = name or config.get("_name_or_path") or c.get("model_type", "model")
        not_full = ((c.get("sliding_window") and c.get("use_sliding_window", True))
                    or any(c.get(k) for k in _NOT_FULL)
                    or any(t != "full_attention" for t in c.get("layer_types") or []))
        if strict and not_full:
            raise NotImplementedError(f"{name}: not every layer keeps a full KV cache; pass strict=False "
                                      "for an upper bound, or build a ModelSpec by hand")
        dtype = (c.get("torch_dtype") or c.get("dtype") or "bfloat16") if kv_dtype == "auto" else kv_dtype
        dtype = str(dtype).lower().removeprefix("torch.")
        dtype_bytes = _DTYPE_BYTES.get(dtype)
        if dtype_bytes is None:
            raise ValueError(f"unknown kv_dtype {dtype!r}; one of {sorted(_DTYPE_BYTES)}")

        L, H, D = c["num_hidden_layers"], c["num_attention_heads"], c["hidden_size"]
        if c.get("kv_lora_rank"):  # MLA (DeepSeek): one latent and one RoPE key per token per layer
            r, rope, nope, v, q = (c["kv_lora_rank"], c["qk_rope_head_dim"], c["qk_nope_head_dim"],
                                   c["v_head_dim"], c.get("q_lora_rank"))
            kv = r + rope
            q_params = D * q + q * H * (nope + rope) if q else D * H * (nope + rope)
            attn_params = q_params + D * (r + rope) + r * H * (nope + v) + H * v * D
            attn_flops = 2 * H * (nope + rope + v)
        else:
            kv_heads, head_dim = c.get("num_key_value_heads") or H, c.get("head_dim") or D // H
            kv = 2 * kv_heads * head_dim
            attn_params = 2 * D * H * head_dim + 2 * D * kv_heads * head_dim  # q, o; k, v
            attn_flops = 4 * H * head_dim  # QK^T and AV
        mlp = _mlp_params(c, D, L)
        return cls(name, L * kv * dtype_bytes, num_layers=L, kv_dtype=dtype,
                   active_params=None if mlp is None else L * attn_params + mlp,
                   attn_flops_per_token=L * attn_flops)

    @classmethod
    def from_hf(cls, repo_id: str, *, revision: str | None = None, token: str | None = None,
                kv_dtype: str = "auto", strict: bool = True) -> ModelSpec:
        """Download ``config.json`` from the Hugging Face Hub (needs ``prefixsim[hf]``).

        Args:
            repo_id: e.g. ``"Qwen/Qwen3-Coder-30B-A3B-Instruct"``.
            revision: a branch, tag or commit; pin it for reproducible results.
            token: for gated models; default ``HF_TOKEN`` from the environment.
            kv_dtype, strict: as for :meth:`from_config`.
        """
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id, "config.json", revision=revision, token=token)
        name = repo_id if revision is None else f"{repo_id}@{revision}"
        return cls.from_config(path, name=name, kv_dtype=kv_dtype, strict=strict)


def _mlp_params(c: dict, D: int, L: int) -> int | None:
    """Gated-MLP parameters a token passes through, summed over layers; None if unknown."""
    dense = 3 * D * c["intermediate_size"] if c.get("intermediate_size") else None
    experts = c.get("num_experts") or c.get("n_routed_experts") or c.get("num_local_experts")
    if not experts:
        return dense and L * dense
    k, size = c.get("num_experts_per_tok"), c.get("moe_intermediate_size") or c.get("intermediate_size")
    if not (k and size):
        return None
    moe = ((k + (c.get("n_shared_experts") or 0)) * 3 * D * size  # routed + DeepSeek's shared
           + 3 * D * (c.get("shared_expert_intermediate_size") or 0)  # Qwen2-MoE's shared
           + D * experts)  # router
    # which layers are MoE: Qwen's mlp_only_layers / decoder_sparse_step, DeepSeek's first_k_dense_replace / moe_layer_freq
    dense_only = set(c.get("mlp_only_layers") or [])
    step, first, freq = c.get("decoder_sparse_step") or 1, c.get("first_k_dense_replace") or 0, c.get("moe_layer_freq") or 1
    n_moe = sum(i not in dense_only and (i + 1) % step == 0 and i >= first and i % freq == 0 for i in range(L))
    if n_moe < L and dense is None:
        return None
    return n_moe * moe + (L - n_moe) * (dense or 0)
