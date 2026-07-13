# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""RSL-RL agent configurations for Unitree Go2 navigation tasks.

The policy block is intentionally identical to ``B2WNavPPORunnerCfg`` so that a
B2W checkpoint can be warm-started via ``--resume`` / ``--load_run`` with
``strict=True`` state-dict loading.
"""

from isaaclab.utils import configclass

from isaaclab_nav_task.navigation.config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2NavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Go2 navigation, tuned for fine-tuning from a B2W checkpoint."""

    num_steps_per_env = 16
    max_iterations = 5000
    save_interval = 200
    # Use TensorBoard by default; wandb requires WANDB_API_KEY and an interactive
    # tty for first-time login, neither of which is available inside the headless
    # docker container. All downstream cfgs (Dev / PureMaze) already override to
    # tensorboard; the base now matches so the mixed-terrain task
    # ``Isaac-Nav-PPO-Go2-v0`` no longer crashes on cold start.
    logger = "tensorboard"
    seed = 42
    wandb_project = "isaaclab_nav_go2"
    experiment_name = "go2_navigation_ppo_ft_from_b2w"
    empirical_normalization = False
    reward_shifting_value = 0.05

    # IMPORTANT: must match the B2W policy block byte-for-byte so that
    # ActorCriticSRU.load_state_dict(..., strict=True) succeeds.
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
        init_noise_std=0.5,                       # ↓ vs 1.0: warm-start, less exploration noise
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        num_cameras=1,
        image_input_dims=(64, 5, 8),
        height_input_dims=(64, 7, 7),
    )

    # Conservative fine-tuning hyperparameters: smaller LR, tighter clip,
    # smaller KL target, lower grad-norm cap.
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.05,
        use_clipped_value_loss=True,
        clip_param=0.1,
        value_clip_param=0.1,
        entropy_coef=0.001,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )


@configclass
class Go2NavPPORunnerDevCfg(Go2NavPPORunnerCfg):
    """Dev cfg used by the smoke-test task (Isaac-Nav-PPO-Go2-Dev-v0).

    Unlike the production ``Go2NavPPORunnerCfg`` (whose hyperparameters are
    deliberately conservative for warm-starting from a B2W checkpoint), the dev
    cfg is meant for *from-scratch* Go2 training with the Odin1 camera. The
    conservative FT settings (entropy_coef=0.001, desired_kl=0.005, clip=0.1,
    lr=3e-4) caused exploration to collapse (noise_std plateaued at ~0.157) and
    success to stall at ~0.5 by 23k iters. Here we re-open exploration and the
    learning rate so the policy can escape that local optimum.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "go2_navigation_ppo_dev"
        self.logger = "tensorboard"

        # ---- Phase 1 (from-scratch, 0->2000): re-open exploration ----
        # init_noise_std only takes effect on a from-scratch run (a resumed
        # run loads the std from the checkpoint, which was ~0.76 after Phase 1).
        # Phase 1 broke the 0.5 plateau (success ~0.65, noise_std held at 0.76),
        # but two issues appeared: (a) adaptive LR floored at 1e-5 because
        # desired_kl=0.01 was too tight against clip=0.2/entropy=0.005, throttling
        # late updates; (b) Loss/value_function spiked to ~127 around iter 1600.
        #
        # ---- Phase 2 (resume 2000->5000): refine & stabilise ----
        # Relax desired_kl so adaptive LR can climb off the 1e-5 floor, anneal
        # exploration down, and tighten clip to curb the value-loss instability.
        #
        # ---- Phase 3 (resume 7000->14000, difficulty [0.2, 0.6]): ----
        # Kept Phase 2 hyperparams. Result: success 0.93, base_contact -71%,
        # tip-over -76%. Confirmed depth is used and avoidance generalises when
        # the task forces it.
        #
        # ---- Phase 4 (resume 14000->24000, difficulty [0.3, 0.8]): ----
        # noise_std fell to ~0.20 in Phase 3 (close to collapse). The Phase 4
        # difficulty jump (0.6 -> 0.8) demands fresh exploration on unseen
        # terrain, so re-open entropy a notch. Result: success climbed 0 -> 0.88
        # (peak 0.91 @ iter 20519). But noise_std overshot to 0.33 by iter 23999,
        # LR hit 1e-5 floor again, and success retreated from 0.905 -> 0.876.
        #
        # ---- Phase 5 (resume 24000->?, difficulty [0.3, 0.8]): refine ----
        # Anneal entropy back to let noise_std settle ~0.22-0.25 so the policy
        # can stop trembling and finish the last refinement past 0.91 success.
        self.policy.init_noise_std = 1.0          # only matters on a fresh run
        self.algorithm.entropy_coef = 0.003       # 0.005 -> 0.003: anneal exploration for Phase 5 refinement
        self.algorithm.desired_kl = 0.02          # 0.01 -> 0.02: let adaptive LR recover off the 1e-5 floor
        self.algorithm.clip_param = 0.15          # 0.2 -> 0.15: smaller, more stable policy updates
        self.algorithm.value_clip_param = 0.15    # 0.2 -> 0.15: match, curb value-loss spikes
        self.algorithm.learning_rate = 1.0e-3     # adaptive ceiling reference
        self.algorithm.num_learning_epochs = 5    # keep: more updates per batch
        self.algorithm.max_grad_norm = 0.8        # 1.0 -> 0.8: tighten to damp the value-fn spike


@configclass
class Go2NavPPORunnerMixedCfg(Go2NavPPORunnerDevCfg):
    """Cold-start hyperparameters for mixed-terrain training (maze + non_maze
    + pits) on the full-size ``Isaac-Nav-PPO-Go2-v0`` task.

    Inherits from the Dev cfg (which already overrides the B2W-FT defaults to
    something usable from scratch), but rolls the *exploration-heavy* knobs
    BACK to the Phase 1/2 values that were proven to break the 0.5 plateau in
    the dev curriculum, rather than the Phase 5 refinement settings the Dev cfg
    currently sits at.

    Distilled lessons from the dev curriculum (see comments in
    ``Go2NavPPORunnerDevCfg``):
      * Phase 1 (entropy=0.005, init_std=1.0): broke success 0.5 -> 0.65 plateau
        but kl=0.01 was too tight (LR floored at 1e-5).
      * Phase 2 (kl=0.02, clip=0.15, epochs=5): unflored the LR and tamed the
        Loss/value_function spike.
      * Phase 5 (entropy=0.003): refinement-only, kills exploration too early
        for a from-scratch run.

    Cold-start recipe = Phase 1 exploration + Phase 2 stability:
      entropy=0.005, init_std=1.0, kl=0.02, clip=0.15, lr=1e-3, epochs=5,
      grad_norm=0.8.
    """

    def __post_init__(self):
        super().__post_init__()
        # Independent experiment dir keeps mixed runs out of the dev TB tree.
        self.experiment_name = "go2_navigation_ppo_mixed"

        # Override Dev's Phase 5 refinement back to cold-start values.
        self.policy.init_noise_std = 1.0          # Phase 1: re-open exploration
        self.algorithm.entropy_coef = 0.005       # Phase 1/4: cold-start exploration
        self.algorithm.desired_kl = 0.02          # Phase 2: avoid 1e-5 LR floor
        self.algorithm.clip_param = 0.15          # Phase 2: avoid value-loss spikes
        self.algorithm.value_clip_param = 0.15
        self.algorithm.learning_rate = 1.0e-3     # adaptive ceiling
        self.algorithm.num_learning_epochs = 5
        self.algorithm.max_grad_norm = 0.8


@configclass
class Go2NavPPORunnerPureMazeCfg(Go2NavPPORunnerDevCfg):
    """Hyperparameters for the pure-maze training task.

    Inherits the Phase 5 dev hyperparams (they shipped a strong checkpoint at
    iter 27800 on the mixed terrain). Only the log directory differs so
    pure-maze runs don't mix with the dev curriculum's TensorBoard history.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_navigation_ppo_puremaze"
