# 🤖 SRU Navigation Training with Isaac Sim

**Reproducible training pipeline for SRU (Spatially-Enhanced Recurrent Units) navigation policy, ported from B2W (wheeled-legged + ZedX depth) to Unitree Go2 (quadruped + LiDAR-derived depth).**

Built on NVIDIA Isaac Sim 4.5.0 + IsaacLab v2.1.1, fully containerized with Docker.

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-76B900?logo=nvidia)](https://developer.nvidia.com/isaac-sim)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-v2.1.1-blue)](https://github.com/isaac-sim/IsaacLab)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)

---

## ✨ Highlights

- **Full B2W → Go2 port** — assets, cameras, terrain, rewards, termination conditions
- **4-stage training pipeline** — Smoke → Cold-start (1024 envs, 30k iter) → Curriculum continuation → PureMaze refinement
- **Evaluation toolchain** — termination rate stats, baseline comparison, collision analysis, depth ablation
- **Env-var driven config** — difficulty, terrain, GPU selection, seed — no code edits needed
- **Multi-GPU parallel training** — independent seeds across GPUs

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container (~30GB)                  │
│                                                             │
│  ┌───────────────┐   ┌──────────────────────────────────┐  │
│  │  Isaac Sim    │   │         IsaacLab v2.1.1           │  │
│  │   4.5.0       │   │  ┌────────────┐ ┌─────────────┐  │  │
│  └───────┬───────┘   │  │ Nav Task   │ │  rsl_rl     │  │  │
│          │            │  │ (Go2 Port) │ │ (SRU Fork)  │  │  │
│          ▼            │  └─────┬──────┘ └──────┬──────┘  │  │
│  ┌───────────────┐   │        │                │         │  │
│  │  PhysX / GPU  │   │        ▼                ▼         │  │
│  │  Simulation   │◄──┤  ┌──────────────────────────┐     │  │
│  └───────────────┘   │  │   PPO + ActorCriticSRU   │     │  │
│                       │  └──────────────────────────┘     │  │
│                       └──────────────────────────────────┘  │
│                                                             │
│  Depth → VAE Encoder (64×5×8) → SRU Policy → SE2 cmd      │
│                          ↓                                  │
│              policy_go2_jit.pt → Joint Actions               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| NVIDIA GPU | 12 GB VRAM | 24 GB+ |
| NVIDIA Driver | ≥ 570 | ≥ 570 (⚠️ avoid 595.x) |
| Docker | 24.0+ | Latest |
| NVIDIA Container Toolkit | Installed | Latest |
| Disk Space | 80 GB | 120 GB+ |
| NGC Access | Required | For `nvcr.io/nvidia/isaac-sim:4.5.0` |

### VRAM Reference

| GPU | NUM_ENVS | GO2_CELL_SIZE |
|---|---|---|
| RTX 5070 12 GB | 768–1024 | 1.0 |
| RTX 4090 24 GB | 2048–2820 | 1.0 / 2.0 |
| A100 80 GB | 8000+ | 1.0 / 2.0 |

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/<your-org>/sru-nav-docker.git
cd sru-nav-docker
cp .env.example .env
# Edit .env — set NGC credentials, GPU index, paths
```

### 2. Build the Image

```bash
docker compose build   # ~30 GB, takes 20-40 min on first build
```

### 3. Launch Container

```bash
docker compose up -d
docker compose exec sru-nav bash
```

### 4. Smoke Test (~30 min)

```bash
TASK=Isaac-Nav-PPO-Go2-Dev-v0 NUM_ENVS=24 MAX_ITER=300 \
GO2_DIFFICULTY="0.0,0.4" GO2_INIT_LEVEL=0 \
./scripts/train_go2_scratch.sh
```

If training starts and loss decreases — you're good to go.

---

## 🏋️ Training Pipeline

### Stage 1 — Cold Start (36–48h)

```bash
TASK=Isaac-Nav-PPO-Go2-v0 NUM_ENVS=1024 MAX_ITER=30000 \
GO2_DIFFICULTY="0.0,0.4" GO2_INIT_LEVEL=0 GO2_CELL_SIZE=1.0 \
SEED=42 RUN_NAME=mixed_scratch_v1 \
./scripts/train_go2_scratch.sh
```

### Stage 2 — Curriculum Continuation

```bash
./scripts/continue_go2.sh --iters 10000 --from-iter 7400
```

### Stage 3 — PureMaze Refinement

```bash
TASK=Isaac-Nav-PPO-Go2-PureMaze-v0 NUM_ENVS=1024 MAX_ITER=10000 \
./scripts/train_go2_scratch.sh
```

### Evaluation

```bash
# Termination rate analysis
PLAY_DIFFICULTY="0.3,0.8" ./scripts/eval_terminations.sh --run-dir <run> --from-iter <iter>

# Baseline comparison
./scripts/compare_baselines.sh

# Collision analysis
./scripts/analyze_collisions.sh
```

### Monitoring

```bash
# TensorBoard
docker compose exec sru-nav ./isaaclab.sh -p -m tensorboard.main --logdir logs --bind_all

# Or use Weights & Biases (configure WANDB_API_KEY in .env)
```

---

## ⚙️ Configuration

All training behavior is controlled via environment variables — no code changes needed.

| Variable | Type | Default | Description |
|---|---|---|---|
| `TASK` | str | `Isaac-Nav-PPO-Go2-Dev-v0` | Gym task ID |
| `NUM_ENVS` | int | `24` | Parallel environment count |
| `MAX_ITER` | int | `1000` | Max PPO iterations |
| `SEED` | int | `42` | Random seed |
| `GPU` | str | — | GPU device index |
| `RUN_NAME` | str | auto | Run directory name |
| `GO2_DIFFICULTY` | `"lo,hi"` | cfg default | Difficulty range |
| `GO2_INIT_LEVEL` | int | `5` | Initial max terrain row |
| `GO2_CELL_SIZE` | float | `1.0` | Terrain tile cell size (m) |
| `PLAY_DIFFICULTY` | `"lo,hi"` | — | Eval-time difficulty override |

### Gym Tasks

| Task ID | Purpose | Typical envs |
|---|---|---|
| `Isaac-Nav-PPO-Go2-v0` | Cold-start training (mixed terrain) | 1024–2048 |
| `Isaac-Nav-PPO-Go2-Dev-v0` | Smoke test / debugging | 24–32 |
| `Isaac-Nav-PPO-Go2-PureMaze-v0` | Stage-2 maze refinement | 1024 |
| `Isaac-Nav-PPO-Go2-Play-v0` | Playback & evaluation | 20 |

---

## 📁 Project Structure

```
sru-nav-docker/
├── Dockerfile                  # Image build recipe
├── docker-compose.yml          # Container runtime config
├── .env.example                # Environment variable template
├── scripts/                    # Train / continue / play / eval / diagnostics
│   ├── train_go2_scratch.sh
│   ├── continue_go2.sh
│   ├── eval_terminations.sh
│   ├── compare_baselines.sh
│   └── ...
├── mount/
│   ├── sru-navigation-sim/     # Task definitions + Go2 port
│   │   ├── isaaclab_nav_task/  # Env, rewards, terrain, obs
│   │   └── scripts/            # Entry points
│   └── rsl_rl/                 # rsl_rl SRU fork (ActorCriticSRU, PPO)
├── IsaacLab/                   # IsaacLab v2.1.1 source
├── assets/baselines/           # Pre-trained baseline checkpoints
├── outputs/logs/               # Training artifacts (TB + ckpt)
└── docs/handover/              # Handover documentation
```

---

## 🔑 Key Dependencies

| Dependency | Source |
|---|---|
| Isaac Sim 4.5.0 | `nvcr.io/nvidia/isaac-sim:4.5.0` (NGC login required) |
| IsaacLab v2.1.1 | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) |
| rsl_rl (SRU fork) | [leggedrobotics/sru-navigation-learning](https://github.com/leggedrobotics/sru-navigation-learning) |
| Nav Task | [leggedrobotics/sru-navigation-sim](https://github.com/leggedrobotics/sru-navigation-sim) |

---

## 📖 Citation

If you use this work, please cite the original SRU navigation paper:

```bibtex
@article{yang2025sru,
  author = {Yang, Fan and Frivik, Per and Hoeller, David and Wang, Chen and Cadena, Cesar and Hutter, Marco},
  title = {Spatially-enhanced recurrent memory for long-range mapless navigation via end-to-end reinforcement learning},
  journal = {The International Journal of Robotics Research},
  year = {2025},
  doi = {10.1177/02783649251401926},
  url = {https://doi.org/10.1177/02783649251401926}
}
```

---

## 🙏 Acknowledgments

- **[ETH Legged Robotics](https://rsl.ethz.ch/)** — SRU navigation (original paper & codebase)
- **[NVIDIA Isaac Sim / IsaacLab](https://developer.nvidia.com/isaac-sim)** — Simulation platform
- **[Unitree Robotics](https://www.unitree.com/)** — Go2 quadruped platform

---

## 📄 License

See [LICENSE](./LICENSE) in the repository root.  
Sub-components (IsaacLab, rsl_rl) retain their respective licenses.
