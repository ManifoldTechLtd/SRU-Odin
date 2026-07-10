#!/usr/bin/env python3
"""Export a trained SRU navigation policy checkpoint to ONNX.

This script reconstructs the ActorCriticSRU model from the checkpoint's
weight shapes (no Isaac Sim / IsaacLab needed), loads the weights, and
calls the built-in export_onnx method.

Usage:
    # Export a specific checkpoint
    python scripts/export_onnx.py --checkpoint outputs/logs/rsl_rl/go2_navigation_ppo_mixed/<run>/model_4600.pt

    # Export and also save JIT
    python scripts/export_onnx.py --checkpoint <path> --jit

    # Override output location
    python scripts/export_onnx.py --checkpoint <path> --output-dir ./exported

Requirements:
    - torch
    - The rsl_rl fork must be importable (pip-installed or on PYTHONPATH)

ONNX model interface (ActorCriticSRU, single camera):
    Inputs:
        - obs:    (1, num_actor_obs)       — flattened observation vector
        - h_in:   (num_layers, 1, hidden)  — LSTM hidden state (zeros at episode start)
        - c_in:   (num_layers, 1, hidden)  — LSTM cell state  (zeros at episode start)
    Outputs:
        - actions: (1, num_actions)
        - h_out:   (num_layers, 1, hidden) — updated hidden state
        - c_out:   (num_layers, 1, hidden) — updated cell state
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def infer_config_from_state_dict(sd: dict) -> dict:
    """Infer ActorCriticSRU constructor args from checkpoint weight shapes."""

    # --- image_input_dims and num_cameras ---
    # attn_image_net cross-attention query projection weight: (Q, image_dim)
    # The first linear in CrossAttentionFuseModule is the query proj.
    # We look for attn_image_net.0.weight or similar.
    attn_keys = [k for k in sd if k.startswith("attn_image_net.") and "weight" in k]
    if not attn_keys:
        raise ValueError("Cannot find attn_image_net weights in checkpoint")
    # image_dim = out_features of the query projection
    # CrossAttentionFuseModule: q_proj weight shape = (image_dim, image_dim)
    # Find the first weight (query proj)
    attn_w = sd[attn_keys[0]]
    image_dim = attn_w.shape[0]  # output features

    # num_image_features = image_dim * H * W
    # We can get it from the actor input: actor_proprioceptive_input_dim + num_image_features = num_actor_obs
    # But we need H, W separately. Get from attn_image_net spatial reshape.
    # CrossAttentionFuseModule stores no spatial dims as parameters.
    # Instead, infer from total image features: look at the first linear of actor
    # which takes mlp_input_dim_actor = actor_proprioceptive_input_dim + image_dim
    # num_image_features = image_dim * H * W
    # We need H, W from image_input_dims=(C, H, W) where C=image_dim.
    # Check if there's a pattern in the attn weights that reveals spatial dims.
    # Fallback: use the known Go2 config defaults.
    # The Go2 config uses image_input_dims=(64, 5, 8), height_input_dims=(64, 7, 7).
    # image_dim=64, H=5, W=8 -> num_image_features = 64*5*8 = 2560
    # We can infer num_image_features from the observation dimension:
    #   num_actor_obs = actor_proprioceptive_input_dim + num_image_features * num_cameras
    # actor_proprioceptive_input_dim = mlp_input_dim_actor - image_dim
    # mlp_input_dim_actor = memory_a input size = rnn input_size

    # Get rnn input_size from memory_a.rnn weight
    rnn_keys = [k for k in sd if k.startswith("memory_a.") and "weight_ih" in k]
    if not rnn_keys:
        raise ValueError("Cannot find memory_a.rnn.weight_ih in checkpoint")
    rnn_input_size = sd[rnn_keys[0]].shape[1]  # input size of LSTM
    rnn_hidden_size = sd[rnn_keys[0]].shape[0] // 4  # LSTM has 4 gates

    # mlp_input_dim_actor = rnn_input_size
    # actor_proprioceptive_input_dim = mlp_input_dim_actor - image_dim
    actor_proprioceptive_input_dim = rnn_input_size - image_dim

    # Get num_image_features from the linear_dropout_actor or actor first layer
    # linear_dropout_actor.linear weight: (actor_hidden_dims[0], rnn_hidden_size)
    ld_keys = [k for k in sd if k.startswith("linear_dropout_actor.") and "weight" in k]
    if ld_keys:
        actor_hidden_0 = sd[ld_keys[0]].shape[0]
    else:
        actor_hidden_0 = 512  # default

    # Get num_actions from actor last layer
    actor_linear_keys = [k for k in sd if k.startswith("actor.") and "weight" in k]
    if not actor_linear_keys:
        raise ValueError("Cannot find actor weights in checkpoint")
    # Last actor layer weight: (num_actions, actor_hidden_dims[-1])
    last_actor_w = sd[actor_linear_keys[-1]]
    num_actions = last_actor_w.shape[0]

    # Infer num_cameras: num_actor_obs = actor_proprioceptive_input_dim + num_image_features * num_cameras
    # We need num_actor_obs. We can get it from the attn_image_net input.
    # CrossAttentionFuseModule q_proj input = image_dim, k_proj input = info_dim = actor_proprioceptive_input_dim
    # But we still can't directly get num_image_features.
    # However, for Go2 with 1 camera: num_image_features = image_dim * H * W = 64 * 5 * 8 = 2560
    # For 2 cameras: num_actor_obs = actor_proprioceptive + 2 * num_image_features
    # We can try both and see which gives a "reasonable" obs dim.
    # Actually, we can get num_image_features from the attn_image_net more carefully.
    # CrossAttentionFuseModule reshapes image (B, C, H, W) -> (B, H*W, C) then projects.
    # The number of spatial tokens = H * W. This doesn't show up as a weight shape.
    #
    # Strategy: use known Go2 defaults (image_input_dims=(64,5,8), num_cameras=1)
    # and verify consistency.
    H, W = 5, 8  # Go2 default
    num_image_features = image_dim * H * W
    num_cameras = 1  # Go2 default

    num_actor_obs = actor_proprioceptive_input_dim + num_image_features * num_cameras

    # Infer height_input_dims for critic (not needed for actor export but for constructor)
    height_dim = image_dim  # default 64
    hH, hW = 7, 7  # Go2 default
    num_height_features = height_dim * hH * hW

    # critic_proprioceptive_input_dim = num_critic_obs - num_height_features - num_image_features * num_cameras - 1
    # We need num_critic_obs. Get from critic rnn input.
    critic_rnn_keys = [k for k in sd if k.startswith("memory_c.") and "weight_ih" in k]
    if critic_rnn_keys:
        critic_rnn_input = sd[critic_rnn_keys[0]].shape[1]
        critic_proprioceptive = critic_rnn_input - image_dim - height_dim
        num_critic_obs = critic_proprioceptive + num_height_features + num_image_features * num_cameras + 1
    else:
        num_critic_obs = num_actor_obs  # fallback

    # rnn_num_layers
    rnn_num_layers = len([k for k in sd if k.startswith("memory_a.") and "weight_ih_l" in k])
    if rnn_num_layers == 0:
        rnn_num_layers = 1

    # actor_hidden_dims from actor weights
    actor_hidden_dims = []
    for k in sorted(actor_linear_keys):
        w = sd[k]
        actor_hidden_dims.append(w.shape[1])
    # Last layer's shape[0] is num_actions, not a hidden dim
    if actor_hidden_dims:
        actor_hidden_dims[-1] = actor_hidden_dims[-1]  # keep last hidden
    # Actually actor weights: layer0 (h1, in), layer1 (h2, h1), ..., layerN (num_actions, hN)
    # So hidden_dims = [in_of_first] ... no, hidden_dims = [h1, h2, ..., hN]
    # weight shapes: (h1, in), (h2, h1), ..., (num_actions, hN)
    # hidden_dims = [out of each layer except last] = [h1, h2, ..., hN]
    actor_hidden_dims = []
    for i, k in enumerate(sorted(actor_linear_keys)):
        w = sd[k]
        if i < len(actor_linear_keys) - 1:
            actor_hidden_dims.append(w.shape[0])  # output dim = next hidden
        # last layer: shape[0] = num_actions

    config = {
        "num_actor_obs": num_actor_obs,
        "num_critic_obs": num_critic_obs,
        "num_actions": num_actions,
        "actor_hidden_dims": actor_hidden_dims if actor_hidden_dims else [512, 256, 128],
        "critic_hidden_dims": actor_hidden_dims if actor_hidden_dims else [512, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
        "image_input_dims": (image_dim, H, W),
        "height_input_dims": (height_dim, hH, hW),
        "rnn_type": "lstm_sru",
        "dropout": 0.2,
        "rnn_hidden_size": rnn_hidden_size,
        "rnn_num_layers": rnn_num_layers,
        "time_embed_dim": 8,
        "num_cameras": num_cameras,
    }

    return config


def main():
    parser = argparse.ArgumentParser(description="Export SRU navigation policy checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <checkpoint_dir>/export)")
    parser.add_argument("--filename", type=str, default="policy.onnx",
                        help="Output filename (default: policy.onnx)")
    parser.add_argument("--jit", action="store_true",
                        help="Also export JIT (policy.pt)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print inferred config and model architecture")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        print(f"[error] Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Load checkpoint
    print(f"[export_onnx] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    print(f"[export_onnx] Checkpoint iteration: {ckpt.get('iter', 'unknown')}")

    # Infer model config from weights
    config = infer_config_from_state_dict(sd)
    if args.verbose:
        print("\n[export_onnx] Inferred config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print()

    # Build model
    from rsl_rl.modules import ActorCriticSRU

    model = ActorCriticSRU(**config)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to("cpu")

    print(f"[export_onnx] Model loaded successfully")
    print(f"  num_actor_obs: {config['num_actor_obs']}")
    print(f"  num_actions:   {config['num_actions']}")
    print(f"  num_cameras:   {config['num_cameras']}")
    print(f"  image_dims:    {config['image_input_dims']}")
    print(f"  rnn_hidden:    {config['rnn_hidden_size']}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent / "export"

    # Export ONNX
    print(f"\n[export_onnx] Exporting ONNX to: {output_dir}/{args.filename}")
    model.export_onnx(path=str(output_dir), filename=args.filename)

    # Optionally export JIT
    if args.jit:
        jit_filename = args.filename.replace(".onnx", ".pt")
        if jit_filename == args.filename:
            jit_filename = "policy.pt"
        print(f"\n[export_onnx] Exporting JIT to: {output_dir}/{jit_filename}")
        model.export_jit(path=str(output_dir), filename=jit_filename)

    print(f"\n[export_onnx] Done! Output: {output_dir}")


if __name__ == "__main__":
    main()
