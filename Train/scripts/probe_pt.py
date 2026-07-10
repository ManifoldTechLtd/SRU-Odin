"""Diagnose a TorchScript locomotion .pt file's interface.

Run inside the sru-nav container; mount this script at /tmp/probe_pt.py.

Reports:
  1. TorchScript graph code (top-level forward signature)
  2. State-dict key names + shapes (gives clues about layer order / obs slicing)
  3. Tries common observation dimensions to find the input width
  4. Output dimension (= number of joint targets the policy emits)
  5. A quick "what zero input produces" check (sanity for output range)
"""

import os
import sys
import torch

DEFAULT_PT = (
    "/workspace/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/"
    "navigation/assets/data/Policies/locomotion/go2/policy_go2.pt"
)
PT_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PT

assert os.path.isfile(PT_PATH), f"File not found: {PT_PATH}"
print(f"[probe] loading: {PT_PATH}")
print(f"[probe] file size: {os.path.getsize(PT_PATH) / 1e6:.2f} MB")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[probe] device: {device}")

is_torchscript = False
model = None
raw_obj = None
try:
    model = torch.jit.load(PT_PATH, map_location=device)
    model.eval()
    is_torchscript = True
    print(f"[probe] format: TorchScript ✓")
    print(f"[probe] type: {type(model).__name__}")
except Exception as e:
    print(f"[probe] torch.jit.load FAILED: {str(e).splitlines()[0][:120]}")
    print(f"[probe] → falling back to torch.load() (pickle)")
    try:
        raw_obj = torch.load(PT_PATH, map_location=device, weights_only=False)
        print(f"[probe] format: pickle (torch.save) — NOT TorchScript")
        print(f"[probe] top-level type: {type(raw_obj).__name__}")
    except Exception as e2:
        print(f"[probe] torch.load ALSO failed: {e2}")
        sys.exit(1)

if not is_torchscript:
    print("\n=========================================================")
    print("  ⚠ This file is NOT a TorchScript model.")
    print("    sru-navigation-sim calls `torch.jit.load(.pt)` which will fail.")
    print("    You need to either:")
    print("      (a) Re-export your policy with torch.jit.script(model).save(...)")
    print("      (b) Provide the original nn.Module class so we wrap+script it")
    print("=========================================================\n")

    # Inspect the loaded object
    print("=========================================================")
    print(" Contents of torch.load() output")
    print("=========================================================")
    if isinstance(raw_obj, dict):
        print(f"  dict with {len(raw_obj)} top-level keys:")
        for k, v in raw_obj.items():
            t = type(v).__name__
            extra = ""
            if isinstance(v, torch.Tensor):
                extra = f" shape={tuple(v.shape)} dtype={v.dtype}"
            elif isinstance(v, dict):
                extra = f" ({len(v)} sub-keys)"
            print(f"    [{k!r}] {t}{extra}")
        # If it has 'model_state_dict' or 'state_dict' or is itself a state_dict, dump shapes
        sd = None
        for key in ["model_state_dict", "state_dict", "model", "policy"]:
            if key in raw_obj and isinstance(raw_obj[key], dict):
                sd = raw_obj[key]
                print(f"\n  → Found state-dict under key {key!r}")
                break
        if sd is None and all(isinstance(v, torch.Tensor) for v in raw_obj.values()):
            sd = raw_obj
            print(f"\n  → Top-level dict appears to be a state_dict itself")
        if sd is not None:
            print(f"\n  State-dict keys & shapes ({len(sd)} entries):")
            for i, (k, v) in enumerate(sd.items()):
                shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
                print(f"    [{i:2d}] {k:60s} {shape}")
                if i >= 50:
                    print(f"    ... ({len(sd) - 51} more)")
                    break
            # First Linear's in_features
            first_linear_in = None
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and v.ndim == 2 and "weight" in k.lower():
                    first_linear_in = v.shape[1]
                    print(f"\n  → first 2D-weight key {k!r} has in_features = {first_linear_in}")
                    print(f"    most likely D_obs = {first_linear_in}")
                    break
            # Last Linear's out_features
            last_linear_out = None
            last_linear_key = None
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and v.ndim == 2 and "weight" in k.lower():
                    last_linear_out = v.shape[0]
                    last_linear_key = k
            if last_linear_out is not None:
                print(f"  → last 2D-weight key {last_linear_key!r} has out_features = {last_linear_out}")
                print(f"    most likely action_dim = {last_linear_out}")
    elif isinstance(raw_obj, torch.nn.Module):
        print(f"  → It's a raw nn.Module instance (untraced). Class: {type(raw_obj).__name__}")
        print(f"  → We can try to forward it directly without TorchScript.")
        model = raw_obj
        model.eval()
        is_torchscript = False
    else:
        print(f"  (unsupported type: {type(raw_obj)})")
    print("\n[probe] done — see message above for next steps.")
    sys.exit(0)

print("\n=========================================================")
print("1. TorchScript .code (forward signature & top-level graph)")
print("=========================================================")
try:
    code = getattr(model, "code", None)
    if code is None:
        print("(no .code attribute -- model may be a ScriptFunction)")
    else:
        print(code[:4000])
except Exception as e:
    print(f"(error reading .code: {e})")

print("\n=========================================================")
print("2. State dict keys & shapes")
print("=========================================================")
sd = model.state_dict()
print(f"total params: {sum(v.numel() for v in sd.values()):,}")
for i, (k, v) in enumerate(sd.items()):
    print(f"  [{i:2d}] {k:60s} {tuple(v.shape)}")
    if i >= 40:
        print(f"  ... ({len(sd) - 41} more)")
        break

# First Linear layer's input dim is a strong hint for D_obs
first_weight = next(
    (v for k, v in sd.items() if v.ndim == 2 and ("weight" in k or k.endswith(".weight"))),
    None,
)
hint_obs = first_weight.shape[1] if first_weight is not None else None
if hint_obs is not None:
    print(f"\n[probe] first Linear weight has in_features = {hint_obs}")
    print(f"        → most likely D_obs = {hint_obs}")

print("\n=========================================================")
print("3. Trying input dimensions (looking for the one that works)")
print("=========================================================")
# Common Go2 obs dims:
#   33 = 3+3+3+3+12+12        (no last_action)
#   45 = 3+3+3+3+12+12+12     (with last_action of joint dim)
#   48 = 3+3+3+3+12+12+12+3   (with last_action including vel cmd)
#   52 = 3+3+3+3+12+12+16     (sru-nav-sim LowLevelPolicyCfg, with B2W-style last_action=16)
#   60 = 3+3+3+3+12+12+12+12  (joint pos+vel+pos+vel duplicated)
#   235 = with height_scan (legged_gym style)
candidate_dims = [33, 36, 39, 42, 45, 48, 51, 52, 60, 96, 235]
if hint_obs is not None and hint_obs not in candidate_dims:
    candidate_dims.insert(0, hint_obs)

working_dims = []
for d in candidate_dims:
    try:
        with torch.inference_mode():
            out = model(torch.zeros(2, d, device=device))
        if isinstance(out, tuple):
            shapes = [tuple(o.shape) for o in out if isinstance(o, torch.Tensor)]
            print(f"  D_obs = {d:3d}  ✓ (tuple output, shapes={shapes})")
        else:
            print(f"  D_obs = {d:3d}  ✓  output shape = {tuple(out.shape)}")
            working_dims.append((d, tuple(out.shape)))
    except Exception as e:
        msg = str(e).split("\n")[0][:90]
        print(f"  D_obs = {d:3d}  ✗  {msg}")

print("\n=========================================================")
print("4. Summary")
print("=========================================================")
if not working_dims:
    print("  ⚠ NO standard input dim worked.")
    print("    → Inspect step-2 first-Linear in_features and try that exact size.")
    print(f"    → Expected: D_obs = {hint_obs}")
else:
    for d, out_shape in working_dims:
        n_out = out_shape[1] if len(out_shape) >= 2 else None
        print(f"  D_obs = {d}, output_dim = {n_out}")
        if n_out == 12:
            print("    → 12 outputs = pure leg joint targets (Go2 has 12 leg joints).")
            print("      You'll use `low_level_velocity_action = None` (the patched action term).")
        elif n_out == 16:
            print("    → 16 outputs matches B2W layout (12 legs + 4 wheels).")
            print("      Adapter must zero-pad if your Go2 only emits 12.")
        elif n_out == 18:
            print("    → 18 outputs = 12 legs + something extra (maybe 6 dummy?).")
        else:
            print(f"    → {n_out} outputs is unusual; verify against your training task.")

print("\n=========================================================")
print("5. Probing zero-input output magnitude (sanity)")
print("=========================================================")
if working_dims:
    d, _ = working_dims[0]
    with torch.inference_mode():
        out = model(torch.zeros(1, d, device=device))
    if isinstance(out, torch.Tensor):
        print(f"  output[0] (first 12 dims): "
              f"{out[0, :12].cpu().numpy().round(3).tolist()}")
        print(f"  output stats: mean={out.mean().item():.3f}, "
              f"std={out.std().item():.3f}, "
              f"min={out.min().item():.3f}, max={out.max().item():.3f}")
        if out.abs().max() > 5.0:
            print("  ⚠ Output magnitude > 5; policy likely emits raw joint angles or torques.")
            print("    Standard IsaacLab convention: tanh-bounded ∈ [-1, 1].")
        elif out.abs().max() < 0.05:
            print("  ⚠ Output extremely small; may be a state_dict-only file (not a working policy).")
        else:
            print("  ✓ Output magnitude looks reasonable.")
print("\n[probe] done")
