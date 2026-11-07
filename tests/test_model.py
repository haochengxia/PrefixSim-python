import json

import pytest

from prefixsim import ModelSpec

# The fields of Qwen/Qwen3-Coder-30B-A3B-Instruct's config.json that matter.
QWEN3_CODER_30B = {
    "model_type": "qwen3_moe",
    "torch_dtype": "bfloat16",
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "sliding_window": None,
    "use_sliding_window": False,
}

# deepseek-ai/DeepSeek-V3: MLA attention, 3 dense layers then MoE.
DEEPSEEK_V3 = {
    "model_type": "deepseek_v3",
    "torch_dtype": "bfloat16",
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "moe_layer_freq": 1,
}


def _ratio(spec, block):
    a, b = map(int, spec.cost_model(block).removeprefix("linear:").split(","))
    return a / b


def test_qwen3_coder():
    spec = ModelSpec.from_config(QWEN3_CODER_30B)
    assert spec.kv_bytes_per_token == 98304 and spec.block_bytes(16) == 1536 * 1024  # 1.5 MiB
    # within 1% of the measured qwen3coder30b_blksz_16 = linear:865,2
    assert _ratio(spec, 16) == pytest.approx(865 / 2, rel=0.01)
    # 4x the tokens: A grows 4x, B 16x
    assert _ratio(spec, 64) == pytest.approx(_ratio(spec, 16) / 4, rel=0.02)
    a, b = map(int, spec.cost_model(16, 10**6).removeprefix("linear:").split(","))
    assert a + b * (10**6 - 1) < 2**31
    assert ModelSpec.from_config(QWEN3_CODER_30B, kv_dtype="fp8").kv_bytes_per_token == 98304 // 2


def test_deepseek_mla():
    spec = ModelSpec.from_config(DEEPSEEK_V3)
    assert spec.kv_bytes_per_token == 61 * (512 + 64) * 2
    assert 30e9 < spec.active_params < 40e9  # ~37B active


def test_refused_and_unknown(tmp_path):
    sliding = dict(QWEN3_CODER_30B, sliding_window=4096, use_sliding_window=True)
    hybrid = dict(QWEN3_CODER_30B, layer_types=["full_attention", "linear_attention"] * 24)
    for config in (sliding, hybrid):
        with pytest.raises(NotImplementedError):
            ModelSpec.from_config(config)
    assert ModelSpec.from_config(sliding, strict=False).kv_bytes_per_token == 98304
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"text_config": QWEN3_CODER_30B}))
    assert ModelSpec.from_config(path).kv_bytes_per_token == 98304
    no_mlp = {k: v for k, v in QWEN3_CODER_30B.items() if "intermediate" not in k}
    with pytest.raises(ValueError, match="active_params"):
        ModelSpec.from_config(no_mlp).cost_model(16)
    with pytest.raises(ValueError, match="kv_dtype"):
        ModelSpec.from_config(QWEN3_CODER_30B, kv_dtype="int3")
