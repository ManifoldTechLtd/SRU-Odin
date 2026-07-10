"""Convert an rsl_rl-style training checkpoint (torch.save dict) into a
deployable TorchScript .pt usable by sru-navigation-sim's torch.jit.load().

Input:  checkpoint with keys {model_state_dict, optimizer_state_dict, iter, infos}
        where model_state_dict has actor.0.weight / actor.2.weight / ...
Output: TorchScript that does forward(obs) -> mean_actions

Usage:
  /workspace/IsaacLab/_isaac_sim/python.sh /tmp/convert_rsl_rl_ckpt_to_jit.py \
      <input_ckpt.pt> <output_jit.pt>
"""

import sys
import torch
import torch.nn as nn


def infer_actor_dims(state_dict):
    """Walk actor.{0,2,4,...}.weight to recover layer sizes & activation slots."""
    layer_dims = []
    indices = sorted(
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("actor.") and k.endswith(".weight")
    )
    for idx in indices:
        w = state_dict[f"actor.{idx}.weight"]
        if w.ndim != 2:
            continue
        if not layer_dims:
            layer_dims.append(w.shape[1])   # in_features of first Linear
        layer_dims.append(w.shape[0])       # out_features
    return layer_dims, indices


def build_actor_mlp(layer_dims, linear_indices, activation_name="elu"):
    """Build an nn.Sequential mirroring rsl_rl's actor:
    Linear -> Activation -> Linear -> Activation -> ... -> Linear (no final activation)."""
    if activation_name.lower() == "elu":
        act_cls = nn.ELU
    elif activation_name.lower() == "relu":
        act_cls = nn.ReLU
    elif activation_name.lower() == "tanh":
        act_cls = nn.Tanh
    else:
        raise ValueError(f"unknown activation: {activation_name}")

    # rsl_rl stores layers at indices 0, 2, 4, ... (activations live in odd slots).
    # We reconstruct that exact layout so load_state_dict by name works.
    layers = []
    n_linears = len(layer_dims) - 1
    for i in range(n_linears):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < n_linears - 1:
            layers.append(act_cls())
    return nn.Sequential(*layers)


class DeployableActor(nn.Module):
    """Wraps the actor MLP and provides a clean forward(obs) -> mean_actions."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def main():
    if len(sys.argv) != 3:
        print("usage: convert_rsl_rl_ckpt_to_jit.py <input.pt> <output.pt>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]

    print(f"[convert] loading checkpoint: {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    sd_full = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # Strip critic / std from state dict for the actor-only deployable model.
    actor_sd = {k: v for k, v in sd_full.items() if k.startswith("actor.")}
    print(f"[convert] actor weights: {len(actor_sd)} tensors")

    layer_dims, linear_indices = infer_actor_dims(sd_full)
    print(f"[convert] actor MLP shape: {' -> '.join(map(str, layer_dims))}")
    print(f"[convert] action_dim = {layer_dims[-1]}, obs_dim = {layer_dims[0]}")

    actor = build_actor_mlp(layer_dims, linear_indices, activation_name="elu")
    # Strip the "actor." prefix so the bare Sequential can load by index keys.
    stripped_sd = {k[len("actor."):]: v for k, v in actor_sd.items()}
    missing, unexpected = actor.load_state_dict(stripped_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"[convert] state_dict loaded ✓")

    deploy = DeployableActor(actor).eval()

    # Sanity forward pass
    with torch.inference_mode():
        fake = torch.zeros(2, layer_dims[0])
        out = deploy(fake)
        print(f"[convert] forward sanity: in={tuple(fake.shape)} -> out={tuple(out.shape)}")
        print(f"[convert] zero-input output[0]: {out[0].numpy().round(3).tolist()}")

    # Trace OR script. We use script since the MLP is straight-line code.
    scripted = torch.jit.script(deploy)
    scripted.save(dst)
    print(f"[convert] saved TorchScript model: {dst}")

    # Verify round-trip
    reloaded = torch.jit.load(dst, map_location="cpu")
    with torch.inference_mode():
        out2 = reloaded(fake)
    assert torch.allclose(out, out2, atol=1e-6), "round-trip mismatch!"
    print(f"[convert] round-trip torch.jit.load() works ✓")


if __name__ == "__main__":
    main()
