"""Function-preserving depth doubling for Qwen3.5 (hybrid linear/full attention).

An inserted layer is made IDENTITY by zeroing its attention output projection
(linear_attn.out_proj or self_attn.o_proj) and its mlp.down_proj -> the residual
passes through unchanged. Interleaving identity copies after each original layer
doubles depth while computing the exact same function at init.
"""
import copy

import torch


def _make_identity(layer):
    """A deep copy of `layer` that is the identity function (zeroed residual writes)."""
    L = copy.deepcopy(layer)
    with torch.no_grad():
        if hasattr(L, "linear_attn"):
            L.linear_attn.out_proj.weight.zero_()
        if hasattr(L, "self_attn"):
            L.self_attn.o_proj.weight.zero_()
        L.mlp.down_proj.weight.zero_()
    return L


def _layers(model):
    return model.model.language_model.layers


def double_depth(model):
    """Interleave an identity copy after every layer: 24 -> 48, function-preserving.
    Returns (model, indices_of_new_layers)."""
    old = list(_layers(model))
    new, new_idx = [], []
    for L in old:
        new.append(L)
        new_idx.append(len(new))      # the identity copy goes right after
        new.append(_make_identity(L))
    model.model.language_model.layers = torch.nn.ModuleList(new)
    cfg = model.model.language_model.config
    cfg.num_hidden_layers = len(new)
    # per-layer config lists are indexed by layer position -> duplicate each entry
    # (an identity copy has the same attention type as the layer it follows)
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = [t for t in cfg.layer_types for _ in range(2)]
    for i, L in enumerate(new):
        for an in ("self_attn", "linear_attn"):
            if hasattr(L, an):
                setattr(getattr(L, an), "layer_idx", i)
    return model, set(new_idx)


def _last_logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits[:, -1, :].float()


def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    name = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to("cuda:0").eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")

    before = _last_logits(m, ids)
    n0 = sum(p.numel() for p in m.parameters())
    m, new_idx = double_depth(m)
    after = _last_logits(m, ids)
    n1 = sum(p.numel() for p in m.parameters())

    diff = (before - after).abs().max().item()
    print(f"layers: 24 -> {len(_layers(m))} ({len(new_idx)} new identity layers)", flush=True)
    print(f"params: {n0/1e9:.3f}B -> {n1/1e9:.3f}B", flush=True)
    print(f"max |logit diff| after doubling: {diff:.6f}", flush=True)
    print(f"FUNCTION-PRESERVING DOUBLING: {'PASS' if diff < 1e-2 else 'FAIL'}", flush=True)
    # sanity: same argmax token
    print("argmax same:", torch.equal(before.argmax(-1), after.argmax(-1)), flush=True)


if __name__ == "__main__":
    main()
