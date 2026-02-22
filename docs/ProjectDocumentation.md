---
id: ProjectDocumentation
aliases: []
tags:
  - project/mlp
  - type/reference
  - status/active
---

# Mario RL Agent - Project Documentation

## Project Overview

This project represents my first hands-on exploration of machine learning through applied practice. Rather than starting with pure theory, I'm building a reinforcement learning agent that can learn to play Super Mario Bros. This practical approach will teach me fundamental ML concepts including neural networks, training pipelines, experiment tracking, and model evaluation—all while working on an engaging, tangible problem.

The ultimate goal extends beyond Mario: I'm building skills and establishing workflows that will transfer to my career goal of applying machine learning to cybersecurity applications, specifically around Suricata rule generation and intelligent incident reporting.

## Project Architecture

### Directory Structure

```
mlp/
├── .claude/              # Claude Code settings
├── configs/              # Hyperparameter configurations (YAML files)
│   ├── dqn_baseline.yaml
│   ├── ppo_baseline.yaml
│   ├── ppo_v2.yaml
│   ├── ppo_v3.yaml
│   └── ppo_v4.yaml      # Frame skip configuration
├── data/                 # Training logs, gameplay videos, episode data
│   ├── logs/
│   └── videos/
├── database/             # SQL schemas, migration scripts for experiment metadata
│   ├── schema.sql
│   └── schema_migration_01.sql
├── docker/               # Dockerfiles for containerized training/deployment (Phase 7)
├── docs/                 # Project documentation and notes (Obsidian vault)
│   ├── daily/           # Daily logs and progress notes
│   ├── templates/       # Note templates
│   ├── ProjectDocumentation.md
│   └── Tasks Dashboard.md
├── .github/
│   └── workflows/       # CI/CD pipelines for automated testing (Phase 7)
├── models/               # Saved model checkpoints and weights
│   ├── checkpoints/
│   ├── dqn_baseline_world1-1_final.zip
│   ├── ppo_v2_world1-1_final.zip
│   └── ppo_v3_world1-1_final.zip
├── notebooks/            # Jupyter notebooks for analysis and exploration
│   ├── 01_environment_exploration.ipynb
│   ├── 02_baseline_vs_dqn_comparison.ipynb
│   └── 03_ppo_vs_dqn_comparison.ipynb
├── scripts/              # Utility scripts for testing and exploration
│   ├── random_agent.py
│   ├── evaluate_model.py
│   └── test_explore_env.py
├── src/                  # Source code
│   ├── environments/    # Gym environment wrappers and preprocessing
│   │   ├── mario_env.py      # Environment factory with wrapper pipeline
│   │   ├── vec_mario_env.py  # Vectorized environments for parallel training
│   │   └── wrappers.py       # Custom wrappers (SkipFrame, Grayscale, etc.)
│   ├── training/        # Training loops and callbacks
│   │   ├── callbacks.py      # W&B and database logging callbacks
│   │   └── train.py          # Main training orchestrator
│   ├── utils/           # Helper functions and utilities
│   │   ├── config_loader.py  # YAML config loading
│   │   └── db_logger.py      # PostgreSQL experiment logging
│   └── __init__.py
├── tests/               # Unit tests for components (Phase 7)
├── CLAUDE.md            # Instructions for Claude Code
├── .gitignore          # Git ignore rules
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── pyproject.toml       # Poetry dependency management
└── README.md            # Project overview and quick start
```

### Key Components

- **Training Pipeline**: Orchestrates the full training workflow from environment initialization through model checkpointing
- **Agent Architecture**: Implements RL algorithms (DQN, PPO) with configurable hyperparameters
- **Environment Wrapper**: Preprocesses game frames and manages observation/action spaces
- **Data Storage**: PostgreSQL database for experiment metadata, hyperparameters, and results
- **Experiment Tracking**: MLflow for model versioning and Weights & Biases for real-time metrics

## Technology Stack

### Core ML Frameworks

- **PyTorch**: Deep learning framework for neural networks
- **Stable-Baselines3**: Production-ready RL algorithm implementations
- **Gymnasium**: Standard RL environment interface
- **gym-super-mario-bros**: NES Mario environment wrapper
- **scikit-learn**: Classical ML algorithms and utilities

### Data & Infrastructure

- **PostgreSQL**: Relational database for experiment tracking
- **SQLAlchemy**: Python ORM for database interactions
- **Pandas**: Data manipulation and analysis
- **DVC**: Data version control for datasets and models
- **MLflow**: Model registry and experiment tracking
- **Weights & Biases**: Real-time training visualization

### Development Tools

- **Poetry**: Python dependency management
- **Docker**: Containerization for reproducible environments
- **GitHub Actions**: CI/CD for automated testing and deployment
- **pytest**: Unit testing framework
- **black**: Code formatting
- **ruff**: Fast Python linting
- **mypy**: Static type checking

### Analysis & Visualization

- **Jupyter Lab**: Interactive notebooks for exploration
- **Plotly**: Interactive visualizations
- **Seaborn**: Statistical graphics
- **TensorBoard**: Training metrics visualization

## Implementation Phases

### Phase 1: Environment Setup ✅ Complete (Dec 26-29, 2025)

- [x] Create project directory structure ✅ 2025-12-26
- [x] Initialize Git repository and GitHub connection ✅ 2025-12-26
- [x] Set up Poetry for dependency management ✅ 2025-12-26
- [x] Configure documentation system (Obsidian) ✅ 2025-12-26
- [x] Install core dependencies (PyTorch, Gymnasium, gym-super-mario-bros) ✅ 2025-12-27
- [x] Set up Python virtual environment ✅ 2025-12-27
- [x] Initialize PostgreSQL database ✅ 2025-12-28
- [x] Create database schema for experiments ✅ 2025-12-29
- [x] Set up Weights & Biases account and project ✅ 2025-12-29
- [x] Configure pre-commit hooks for code quality ✅ 2025-12-29

### Phase 2: Baseline Agent ✅ COMPLETE (Dec 30-31, 2025)

- [x] Install and test gym-super-mario-bros environment ✅ 2025-12-30
- [x] Implement random agent to understand environment mechanics ✅ 2025-12-31
- [x] Build frame preprocessing pipeline: ✅ 2025-12-31
  - [x] Grayscale conversion ✅ 2025-12-31
  - [x] Frame resizing ✅ 2025-12-31
  - [x] Frame stacking (temporal context) ✅ 2025-12-31
  - [x] Normalization ✅ 2025-12-31
- [x] Create first Jupyter notebook for environment exploration ✅ 2025-12-31
- [x] Log baseline experiment to Weights & Biases ✅ 2025-12-31
- [x] Enhanced baseline with 13 comprehensive metrics ✅ 2025-12-31
- [x] Database schema migration for episode metrics ✅ 2025-12-31
- [~] Record and save gameplay videos ⚠️ Deferred (see note below)

**Phase 2 Achievements:**

- Random baseline: avg reward ~380, max x_pos 434, 0/10 level completions
- Enhanced metrics: 13 tracked values (x_pos, score, time, coins, life, status, flag_get, etc.)
- wandb cloud integration with authentication and real-time logging
- PostgreSQL schema extended with 8 new episode metric columns
- Success criteria defined: x_pos > 434, score ≥ 100, flag_get = True

**Video Recording Note:**
Attempted multiple approaches (RecordVideo wrapper, manual imageio frame collection) but discovered gym-super-mario-bros has render_mode='rgb_array' compatibility issues (unmaintained since 2019). Videos created but contain static frames. **Decision: Proceed with metrics-only approach.** Comprehensive wandb tracking provides sufficient baseline proof. Video recording deferred to Phase 3 with alternative approach (render_mode='human' + screen recording).

### Phase 3: Simple RL Algorithm ✅ COMPLETE (Jan 2-4, 2026)

- [x] Learn DQN concepts (Q-learning, experience replay, target networks) ✅ 2026-01-02
- [x] Create YAML configuration system for hyperparameters ✅ 2026-01-02
- [x] Create config loader utility (`src/utils/config_loader.py`) ✅ 2026-01-02
- [x] Simplify action space with JoypadSpace (256 → 7 actions) ✅ 2026-01-02
- [x] Build database logging utilities with connection pooling ✅ 2026-01-02
- [x] Create training script structure (main entry point, argument parsing) ✅ 2026-01-02
- [x] Integrate Stable-Baselines3 DQN with configuration ✅ 2026-01-02
- [x] Add custom callbacks for W&B and database logging during training ✅ 2026-01-02
- [x] Test end-to-end training run (short trial to verify everything works) ✅ 2026-01-03
- [x] Run full DQN training (2M timesteps) ✅ 2026-01-03
- [x] Create evaluation script (load trained model, run test episodes) ✅ 2026-01-04
- [x] Build analysis notebook comparing random vs. trained agent ✅ 2026-01-04

**Phase 3 Progress: 12/12 tasks complete (100%)** ✅ COMPLETE!

**Completed Artifacts:**

- `configs/dqn_baseline.yaml` - Experiment configuration (2M timesteps, CnnPolicy, SIMPLE_MOVEMENT)
- `src/utils/config_loader.py` - YAML configuration loader
- `src/environments/mario_env.py` - Environment helper with simplified actions + CompatibilityWrapper
- `src/environments/wrappers.py` - 5 custom wrappers (Compatibility, Grayscale, Resize, FrameStack, Transpose)
- `src/utils/db_logger.py` - Database logging with connection pooling (5 functions) + metadata tracking
- `src/training/train.py` - Complete training orchestrator with git/version metadata tracking
- `src/training/callbacks.py` - Custom WandbCallback and DatabaseCallback
- `scripts/evaluate_model.py` - Model evaluation script with rendering and statistics ✅ 2026-01-04
- `scripts/random_agent.py` - Updated with database logging ✅ 2026-01-04
- `notebooks/02_baseline_vs_dqn_comparison.ipynb` - Comprehensive analysis notebook ✅ 2026-01-04
- `data/videos/dqn_baseline_evaluation_2026-01-04.mp4` - Trained agent gameplay video ✅ 2026-01-04
- Successful 2M timestep training run (785 episodes, ~12 hours) ✅ 2026-01-03

**Phase 3 Results:**

- **DQN Performance**: 5.33x better reward (360 → 1920), 2.92x further distance (350 → 1024 pixels), 14.76x better score (40 → 590)
- **Training Success**: Agent learned meaningful strategies (rightward movement, enemy interaction, coin collection)
- **Database**: 785 training episodes + 10 baseline episodes logged with 13 comprehensive metrics each
- **Analysis**: Full comparative analysis in Jupyter notebook with interactive visualizations
- **Evaluation**: Working evaluation pipeline with video recording capability

**Key Learnings:**

- DQN fundamentals: Q-function, Bellman equation, bootstrapping, experience replay, target networks
- YAML configuration management for reproducible experiments
- Database connection pooling with defensive programming and type safety
- Simplified action space: SIMPLE_MOVEMENT provides 7 useful actions vs 256 button combinations
- Stable-Baselines3 integration: CnnPolicy, hyperparameter passing, model.learn() API
- Callback pattern: Event hooks for logging during training without modifying SB3 code
- Training pipeline architecture: Orchestrator pattern coordinating config, wandb, database, environment, agent, callbacks
- **Wrapper composition pattern**: Build complex preprocessing from simple, single-purpose wrappers
- **Image format conventions**: PyTorch uses (C,H,W), NumPy/TensorFlow use (H,W,C) - TransposeWrapper bridges gap
- **API compatibility strategies**: CompatibilityWrapper bridges old Gym and new Gymnasium APIs with try/except fallback
- **Metadata tracking for reproducibility**: Git hash, Python version, PyTorch version logged to database
- **Systematic debugging**: Fixed 29 total integration issues (16 training, 13 evaluation) across multiple system boundaries
- **Real ML engineering**: Integration debugging is 50% of the job - tutorials skip this critical skill!
- **Integration testing principles**: Tests must exercise full code paths - 1000 steps missed bugs, 30k steps found them all
- **Type system boundaries**: NumPy → Python → PostgreSQL require explicit type conversions at integration points
- **Schema evolution challenges**: Code and database naming must stay synchronized across phases
- **Process management**: Long-running training requires protection from system power management (hypridle, suspend)
- **Evaluation best practices**: deterministic=False allows stochastic policy with exploration, often performs better than deterministic=True
- **Data analysis workflow**: PostgreSQL → Pandas → Plotly pipeline for comprehensive experiment analysis
- **Partial learning in RL**: Agents can improve significantly (5x reward) without completing the task (0% success rate)
- **Visualization impact**: Interactive plots (Plotly) reveal learning curves and performance trends missed by raw statistics

### Phase 4: PPO Baseline & Comparison ✅ COMPLETE (Jan 10-11, 2026)

- [x] Learn PPO concepts (on-policy, actor-critic, advantage estimation) ✅ 2026-01-10
- [x] Create PPO configuration file (`configs/ppo_baseline.yaml`) ✅ 2026-01-10
- [x] Update training script to support multiple algorithms ✅ 2026-01-10
- [x] Implement vectorized environment wrapper (`src/environments/vec_mario_env.py`) ✅ 2026-01-10
- [x] Add SubprocVecEnv for parallel environment execution (8 envs) ✅ 2026-01-10
- [x] Test PPO training pipeline with short runs (10k, 50k timesteps) ✅ 2026-01-10
- [x] Run full PPO training (2M timesteps) ✅ 2026-01-11 (policy collapsed - see notes)
- [x] Compare PPO vs DQN performance ✅ 2026-01-11 (DQN significantly better)
- [~] Create comparison notebook with visualizations ⚠️ Deferred (no training data logged)
- [x] Document PPO vs DQN learnings ✅ 2026-01-11

**Phase 4 Progress: COMPLETE (with documented failure)**

**Critical Issue Discovered: Policy Collapse**
PPO training ran for 2M steps (~10.2 hours) but the policy collapsed after 800k steps:

- 800k checkpoint: Agent moves right, reaches x=353 (reasonable early learning)
- 1.6M checkpoint: Agent retreats after initial progress
- 2M final: Agent immediately runs backwards into corner (degenerate policy)

**Root Causes Identified:**

1. **Callback bug**: `WandbCallback` and `DatabaseCallback` only checked `dones[0]` - missing 7/8 of episode completions with 8 parallel environments
2. **No early warning**: Without episode metrics, couldn't detect collapse in real-time
3. **Too many parallel envs**: 8 environments at 82-95% CPU caused thermal throttling and longer training time (10.2 hours vs expected 4-5)
4. **Hyperparameters likely too aggressive**: Learning rate 0.0001, entropy 0.01 may have caused instability

**Completed Artifacts:**

- `configs/ppo_baseline.yaml` - PPO experiment configuration
- `src/environments/vec_mario_env.py` - Vectorized environment wrapper using SubprocVecEnv
- `src/training/train.py` - Updated with multi-algorithm support (PPO + DQN)
- `models/ppo_baseline_world1-1_800000_steps.zip` - Best PPO checkpoint (before collapse)
- `models/ppo_baseline_world1-1_final.zip` - Collapsed policy (for reference)

**Key Learnings:**

- **PPO vs DQN architecture**: On-policy (fresh data) vs off-policy (replay buffer)
- **Actor-critic**: PPO learns policy + value function; advantage = Q(s,a) - V(s)
- **Parallel environments**: SubprocVecEnv enables true multiprocessing parallelism
- **CPU overload**: 8 envs at 90%+ CPU was too aggressive - caused thermal throttling
- **PPO training metrics**: approx_kl, clip_fraction, explained_variance indicate training health
- **Policy collapse**: PPO can catastrophically forget learned behavior if training continues too long
- **Vectorized callback bug**: Standard callbacks check only `[0]` index - must iterate over all envs
- **Evaluation determinism**: PPO prefers `deterministic=True` (unlike DQN's `deterministic=False`)
- **Monitoring is critical**: Without proper logging, policy collapse went undetected for hours
- **Value loss spikes**: Periodic spikes to 30-50 were warning signs of instability

### Phase 5: Infrastructure Fixes, Reward Shaping & Hyperparameter Tuning ✅ COMPLETE (Jan 11-17, 2026)

**Part A: Fix Infrastructure (Prerequisites)** ✅ COMPLETE

- [x] Fix callbacks for vectorized environments:
  - [x] Update `WandbCallback` to iterate over all `n_envs` ✅ 2026-01-11
  - [x] Update `DatabaseCallback` to iterate over all `n_envs` ✅ 2026-01-11
  - [x] Test callbacks with short PPO run to verify logging works ✅ 2026-01-11
- [x] Add `VecMonitor` wrapper for episode tracking ✅ 2026-01-11
- [x] Fix `CompatibilityWrapper.step()` for gym/gymnasium API conversion ✅ 2026-01-11
- [x] Reduce parallel environments (8 → 4) to prevent CPU throttling ✅ 2026-01-11
- [x] Verify episode metrics appear in W&B and PostgreSQL ✅ 2026-01-11

**Part B: Hyperparameter Tuning** ✅ COMPLETE

- [x] Lower learning rate: 0.0001 → 0.00003 ✅ 2026-01-11
- [x] Increase entropy coefficient: 0.01 → 0.02 ✅ 2026-01-11
- [x] Created `configs/ppo_v2.yaml` with tuned parameters ✅ 2026-01-11

**Part C: Reward Shaping** ✅ COMPLETE

- [x] Implement `RewardShapingWrapper`:
  - [x] Forward bonus (+0.1 per pixel moved right) ✅ 2026-01-11
  - [x] Backward penalty (-0.1 per pixel moved left) ✅ 2026-01-11
  - [x] Idle penalty (-0.2 per step standing still) ✅ 2026-01-11
  - [x] Death penalty (-50 for losing a life) ✅ 2026-01-11
  - [x] Early termination (episode ends after 150 stuck steps) ✅ 2026-01-11
  - [x] Milestone bonuses (650→+150, 900→+100, 1200→+150, 1600→+200, 2000→+250) ✅ 2026-01-11

**Part D: Full Training Runs** ✅ COMPLETE

- [x] Launch 2M timestep PPO v2 training run ✅ 2026-01-11
- [x] Evaluate PPO v2 trained model ✅ 2026-01-12
- [x] Create comparison notebook (PPO v2 vs DQN baseline) ✅ 2026-01-12
- [x] Clean up PostgreSQL database ✅ 2026-01-12- [x] Generate visualizations ✅ 2026-01-12
- [x] Research successful Mario PPO implementations ✅ 2026-01-12
- [x] Create PPO v3 config with LR scheduler (10M steps) ✅ 2026-01-12
- [x] Launch 10M timestep PPO v3 training run ✅ 2026-01-13
- [x] Evaluate PPO v3 trained model ✅ 2026-01-14
- [x] Update comparison notebook with PPO v3 results ✅ 2026-01-14

**PPO v3 Results - BREAKTHROUGH! 🎉**
| Metric | DQN (2M) | PPO v2 (2M) | PPO v3 (10M) |
|--------|----------|-------------|--------------|
| Avg Distance | 1,024 px | 687 px | **1,319 px** 🏆 |
| Avg Reward | 1,920 | 700 | **2,025** 🏆 |
| Max Distance | 1,673 px | 2,226 px | 1,674 px |
| Episodes | 785 | 2,197 | 4,684 |

**Key Achievement:** PPO v3 finally beat DQN! 1.29x further distance, 1.05x more reward.

**Part E: Frame Skip Optimization** ✅ COMPLETE (Jan 15, 2026)

- [x] Implement `SkipFrameWrapper` (skip=4 frames per action) ✅ 2026-01-15
- [x] Add wrapper to environment pipeline (after CompatibilityWrapper) ✅ 2026-01-15
- [x] Create `configs/ppo_v4.yaml` with frame skip enabled ✅ 2026-01-15
- [x] Integrate frame_skip through full config chain (train.py → vec_mario_env → mario_env) ✅ 2026-01-15
- [x] Run 50k test: Agent **immediately broke 722 barrier!** ✅ 2026-01-15
- [x] 10M training run complete ✅ 2026-01-17
- [x] Evaluate PPO v4 trained model ✅ 2026-01-17
- [x] Update comparison notebook with PPO v4 results ✅ 2026-01-17

**PPO v4 FINAL RESULTS - MASSIVE BREAKTHROUGH! 🚀**

| Metric | DQN (2M) | PPO v3 (10M) | **PPO v4 (10M+Skip)** |
|--------|----------|--------------|----------------------|
| Avg Distance | 1,024 px | 1,319 px | **2,725 px** 🏆 |
| Max Distance | 1,673 px | 1,674 px | **2,757 px** 🏆 |
| Avg Reward | 1,920 | 2,025 | **6,210** 🏆 |
| Episodes | 785 | 4,684 | **18,985** |
| Level Progress | 31% | 40% | **83%** |

**Key Achievement:** PPO v4 reaches 83% of the level consistently! That's 2.07x further than PPO v3 and 2.66x further than DQN.

**New Barrier Identified: ~2,700 Pixels**
- Agent consistently reaches 2,470-2,757 pixels (tight 287px range)
- This indicates a reliable strategy up to a specific obstacle
- Next step: PPO v5 with SpeedrunRewardWrapper for cleaner reward signal

**Early Test Results (50k steps) - BREAKTHROUGH!**
| Episode | Distance | Reward | Notes |
|---------|----------|--------|-------|
| 1 | 722 px | 1,792 | At the pipe |
| 2 | **1,422 px** | 3,043 | Past the pipe! 🎉 |
| 5 | **1,405 px** | 1,740 | Consistent! |
| 11 | **1,425 px** | 2,359 | Breakthrough confirmed! |

**Why Frame Skip Works:**
Without frame skip, the agent processes every frame at 60 FPS:

- To "hold jump" for 0.5 seconds = must output "jump" 30 consecutive times
- This is the core of the "Tall Pipe Problem"

With frame skip (skip=4):

- Each action repeats for 4 frames automatically
- To "hold jump" for 0.5 seconds = only need ~7 consecutive "jump" actions
- **Reduces exploration difficulty by 4x!**

**Frame Skip Implementation Details:**
```python
class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = max(skip, 1)  # Ensure at least 1

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return (obs, total_reward, terminated, truncated, info)
```

**Config Chain:**
`ppo_v4.yaml (frame_skip: 4)` → `train.py` → `make_vec_mario_env(skip=)` → `make_mario_env(skip=)` → `SkipFrameWrapper(env, skip=)`

**Reward Investigation Results:**
During training, noticed suspicious reward values where different distances showed similar rewards. Root cause: base game reward (score/coins) was adding noise to the shaped rewards. Initial solution for v5: `SpeedrunRewardWrapper` completely replaces base game reward with pure progress-based signal. **However, this approach failed** - see "PPO v5 Debugging" section below.

**PPO v2 Results (2M steps):**

- DQN still outperformed: avg 1,024 px vs PPO's 687 px
- Root cause: "Tall Pipe Problem" at x≈700 - agent can't chain enough jumps
- Key insight: Mario's physics require temporal action sequences (holding jump), difficult with discrete actions

**PPO v3 Configuration Changes:**
| Parameter | v2 → v3 | Rationale |
|-----------|---------|-----------|
| total_timesteps | 2M → 10M | Successful implementations trained 5x longer |
| use_lr_scheduler | false → true | Linear annealing (3e-05 → 0) improves stability |
| clip_range | 0.2 → 0.15 | Stanford's successful config |
| n_epochs | 5 → 10 | More data reuse per update |
| max_stuck_steps | 150 → 300 | More attempts at obstacles |

**Phase 5 Artifacts:**

- `configs/ppo_v2.yaml` - Tuned PPO configuration
- `configs/ppo_v3.yaml` - 10M step config with LR scheduler
- `configs/ppo_v4.yaml` - Frame skip configuration (skip=4)
- `configs/ppo_v5.yaml` - SpeedrunRewardWrapper configuration (prepared for next run)
- `models/ppo_v4_world1-1_final.zip` - Best model yet (83% level progress)
- `src/environments/wrappers.py` - `RewardShapingWrapper`, `SkipFrameWrapper`, `SpeedrunRewardWrapper`
- `src/environments/mario_env.py` - Added `skip` parameter, integrated `SkipFrameWrapper`
- `src/environments/vec_mario_env.py` - Added `VecMonitor` wrapping, `skip` parameter passthrough
- `src/training/callbacks.py` - Fixed vectorized environment iteration
- `src/training/train.py` - Added `linear_schedule()`, reads `frame_skip` from config
- `notebooks/03_ppo_vs_dqn_comparison.ipynb` - Full analysis notebook with all 5 agents
- `docs/images/v4mlppics/` - PPO v4 visualization images
- `scripts/analyze_experiment.py` - Database analysis script for reward debugging

**Key Learnings:**

- SubprocVecEnv runs environments in separate processes - print statements don't appear in main terminal
- Reward shaping alone doesn't make episodes end faster - need early termination too
- On-policy algorithms (PPO) need stronger reward signals than off-policy (DQN) because they discard data after each update
- Gym vs Gymnasium API differences: old gym returns 4 values from step(), new gymnasium returns 5
- **Deterministic vs Stochastic evaluation**: `deterministic=False` samples from policy (matches training), `deterministic=True` always picks highest probability
- **Action space limitations**: SIMPLE_MOVEMENT (7 actions) can't "hold" buttons - high jumps require chaining consecutive jump actions
- **Training duration matters**: Research showed successful Mario PPO runs used 10M+ steps with LR scheduling
- **Frame skip is transformative**: Reduces exploration difficulty for temporal action sequences (like holding jump) by the skip factor. Standard practice in Atari/game RL.
- **Wrapper order matters**: SkipFrame should come early in pipeline (after API compatibility, before preprocessing) to avoid wasting computation on skipped frames
- **Defensive programming**: Use `max(skip, 1)` to ensure skip is never 0, preventing division/loop edge cases

**PPO v5 Debugging (Jan 18, 2026) - Reward Engineering Deep Dive:**

Multiple failed runs attempting to use pure custom reward shaping without base game rewards.

**The Oscillation Exploit:**
- SpeedrunRewardWrapper initially set `reward = 0.0` to completely replace base game rewards
- With forward_bonus=1.0 and backward_penalty=0.4, oscillation became profitable:
  - Move right 100px (+100) then left 100px (-40) = +60 reward for staying in place!
- Agents learned to oscillate back and forth farming infinite reward
- Diagnosis: Built `scripts/analyze_experiment.py` to query PostgreSQL for anomalies
- Found episodes with 176 distance getting 7,000+ reward

**Key Reward Engineering Learnings:**

1. **Asymmetric penalties create exploits**: backward_penalty must be ≥ forward_bonus to prevent oscillation arbitrage
2. **Base game rewards encode domain knowledge**: Coins act as breadcrumbs, enemy kills teach engagement strategies. Pure x_delta reward says WHAT is good but not HOW to achieve it.
3. **Reward magnitude affects value function stability**: Large per-step rewards (1.0 vs 0.1) cause high value_loss and training instability
4. **"Learned helplessness"**: If forward progress always leads to death (-200 penalty), agent prefers slow oscillation bleeding over "certain death" from pushing forward
5. **Policy collapse indicators**: entropy_loss approaching 0, repeated exact same distance/reward values, approx_kl > 0.1

**Bug Found:** Clip range scheduler was never applied! Line 153 of train.py used `hyperparams["clip_range"]` instead of the `clip_range` variable.

**PPO v5 Final Results - FAILURE:**
- SpeedrunRewardWrapper created **negative reward/distance correlation** (-0.174)
- Agent learned that going further = less reward
- RIGHT_ONLY action space too restrictive
- Avg distance: 556 px (worse than random!)

**PPO v6 (Jan 23, 2026) - Entropy Collapse:**

Attempted to combine v4's working foundation with v5's clip scheduler.

| Parameter | v6 Value |
|-----------|----------|
| action_space | SIMPLE_MOVEMENT |
| reward_wrapper | standard (RewardShaping) |
| ent_coef | 0.02 → 0.03 (after first failure) |
| clip_range | 0.15 → 0.05 (scheduler) |
| learning_rate | 0.00005 |

**v6 Results - FAILURE:**
- Attempt #1: Killed at 31% (3.1M steps) due to entropy collapse
- Entropy dropped from -1.13 → -0.08 (14x reduction = deterministic policy)
- Agent stuck at ~2007 pixels (700px behind v4)
- Attempt #2 with ent_coef=0.03 also failed

**PPO v7 (Jan 24-26, 2026) - RAM Observations:**

Complete paradigm shift: Instead of pixel observations with CnnPolicy, use RAM-based grid observations with MlpPolicy (based on yumouwei's implementation).

| Parameter | v7 Value |
|-----------|----------|
| observation_mode | ram (13x16 grid from NES RAM) |
| policy | MlpPolicy (vs CnnPolicy) |
| model_size | 1.5MB (vs 21MB for CNN) |
| learning_rate | 0.0003 |
| n_steps | 2048 |
| batch_size | 64 |
| ent_coef | 0.01 |

**v7 Implementation:**
- `SMBGrid` class: Extracts game state from NES RAM into 13x16 grid
- Grid values: 0=empty, 1=tile/block, 2=Mario, -1=enemy
- `RAMObservationWrapper`: Frame stacking for temporal info, flattened for MLP
- Training completed 10M steps (Jan 26, 2026)

**v7 Results - SUCCESS!**

| Metric | v4 (CNN) | v7 (RAM) |
|--------|----------|----------|
| Avg Distance | 1,948 px | **2,321 px** (+19%) |
| Avg Reward | 4,447 | **6,150** (+38%) |
| Max Distance | **3,154 px** | 2,776 px |
| Consistency (stddev) | 893 | **762** |
| Episodes Reaching 2700+ | 42% | **72%** |
| Reward-Distance Correlation | **0.826** | 0.230 |

**Key Findings:**
- v7 is more consistent, v4 has higher variance but occasionally breaks through to 3000+
- The apparent 2760px "ceiling" was later revealed to be a **metrics/callback bug** (see Phase 6)
- v7 completed levels during evaluation but not during training (final weights + luck)
- v7's learning curve was stable from the start (no collapse/recovery like v4)

**Both approaches are valid for Phase 6:**
- v4's strong reward correlation (0.826) provides clearer learning signal
- v7's RAM observations are more interpretable for imitation learning
- Combining both: RAM obs + RewardShapingWrapper + BC pre-training

**Phase 5 Artifacts:**
- `configs/ppo_v2.yaml` through `configs/ppo_v7.yaml`
- `models/ppo_v4_world1-1_final.zip` - Best CNN-based model (83% level progress)
- `models/ppo_v7_world1-1_final.zip` - RAM-based model (1.5MB)
- `src/environments/wrappers.py` - Added `SMBGrid`, `RAMObservationWrapper`
- `src/environments/mario_env.py` - Added `observation_mode` parameter
- `src/training/callbacks.py` - Added multi-stage distance tracking
- `database/schema_migration_02.sql` - Added `max_x_pos`, `final_x_pos` columns

### Phase 6: Evaluation & Fine-tuning ✅ Complete (Feb 2026)

**Original Goal:** Teach the agent speedrunning techniques via imitation learning to break through the 2760px ceiling.

**Actual Outcome:** The "ceiling" was a metrics bug. v7 was already beating 1-1.

#### Part A: Imitation Learning POC — Abandoned

Built `scripts/poc_frame_skip_replay.py` to test whether smbdataset demonstration data could be replayed through the gym environment:

- Downloaded smbdataset (933MB, 280 episodes, 32 levels)
- Identified 5 winning 1-1 episodes with ~2200 frames each
- Implemented action parsing, NES→SIMPLE_MOVEMENT mapping, and replay loop

**POC Results:**
- Full-speed replay (v3 env, frameskip=1): Only reached x=839-899 (25% of level)
- Subsampled (every 4th frame): Dropped to ~35% of full-speed distance
- Root causes: Action timing mismatch between 60fps recordings and env physics, lossy action mapping (NES 256→SIMPLE_MOVEMENT 7)

**Decision:** Imitation learning **not worth the investment**. Raw action sequences are too brittle; a model trained on noisy trajectories would inherit those problems plus add learning noise.

#### Part B: v8 Fine-tuning — The Real Discovery

Pivoted to fine-tuning v7 weights with fixed callbacks:

**v8 Config (`ppo_v8.yaml`):**
- Loads v7 final weights (`pretrained_model` support added to `train.py`)
- Learning rate: 0.0001 (reduced from v7's 0.0003 for fine-tuning)
- 5M additional timesteps
- Same RAM/MlpPolicy/RewardShapingWrapper setup as v7

**Critical Discovery — The 2760px "Ceiling" Was a Metrics Bug:**

With fixed callbacks, v8 training immediately revealed:
- **43% win rate (78/179 episodes)** from the very first rollouts — before any new training
- v7 was already completing World 1-1 consistently
- Episodes with ~904 distance and high rewards (7200-7500) = completed 1-1, died in 1-2
- Episodes with ~2753 distance and lower rewards (6440-6450) = died before finishing 1-1
- The "ceiling" was the callback bug misreporting multi-stage progress, not a training limitation

**Phase 6 Artifacts:**
- `scripts/poc_frame_skip_replay.py` — Frame skip replay POC (complete)
- `smbdataset/` — Downloaded demonstration data (3.4GB extracted)
- `configs/ppo_v8.yaml` — Fine-tuning config with pretrained model loading
- `src/training/train.py` — Added `pretrained_model` support via `PPO.load()`

**Infrastructure Changes:**
- Training moved to `mlp-dev` distrobox (Fedora 43) — resolves nes_py/pyglet X11 issues on ZenaOS/Nix
- PostgreSQL 16 running as podman container (`mario-postgres`) with `mario_pgdata` volume

**Key Lessons:**
1. Always validate metrics before concluding there's a performance plateau
2. Imitation learning from recorded demos has fundamental timing/mapping challenges
3. Fine-tuning existing weights is far more efficient than building new pipelines when the model already works

### Phase 7: Production & Analysis ⏳ NEXT (Mar 2026)

- [ ] Containerize training environment with Docker:
  - [ ] Multi-stage build (training vs. inference)
  - [ ] GPU support configuration
- [ ] Set up GitHub Actions workflows:
  - [ ] Run tests on pull requests
  - [ ] Code quality checks (black, ruff, mypy)
  - [ ] Automated model evaluation
- [ ] Create comprehensive data analysis dashboards:
  - [ ] Training stability analysis
  - [ ] Hyperparameter correlation studies
  - [ ] Performance comparison across algorithms
- [ ] Build model evaluation pipeline:
  - [ ] Standardized test episodes
  - [ ] Statistical significance testing
  - [ ] Performance benchmarking
- [ ] Write comprehensive documentation:
  - [ ] API documentation
  - [ ] Training guides
  - [ ] Architecture decisions
  - [ ] Lessons learned

### Project Complete 🎉

After Phase 7, this project concludes. The skills developed here directly feed into the next learning projects:

**Learning Roadmap:**

```
Mario RL Agent (current)
    ↓ ML fundamentals, training pipelines, experiment tracking
Network Modeling & Fundamentals
    ↓ Understanding network traffic patterns, protocols
Network Attack Vectors
    ↓ Knowledge for testing ML-based detection
Firewall Tool Development
    ↓ Build the core infrastructure
ML-Enhanced Firewall
    └── Apply everything: anomaly detection, intelligent alerting
```

**Skills Transferring to Cybersecurity:**

- **Preprocessing pipelines**: Frame processing → Packet/flow feature extraction
- **Experiment tracking**: W&B/PostgreSQL → Model versioning for detection rules
- **Reward shaping**: Game rewards → Alert severity scoring
- **On-policy vs off-policy**: Understanding when models need fresh data vs historical
- **Deployment (Phase 7)**: Containerization and CI/CD for production ML systems

## Future Applications

Skills and tools developed in this project will directly transfer to cybersecurity applications:

- **ML-Enhanced Suricata Rules**: Apply anomaly detection and pattern recognition to network traffic
- **Intelligent Incident Management**: Use clustering and classification for alert correlation
- **Threat Intelligence**: Automated IOC extraction and threat classification
- **Data Pipeline Experience**: Transfer PostgreSQL, MLflow, and visualization skills to security operations

## Links & Resources

- [GitHub Repository](https://github.com/BrendonJL/mlp)
- [Project Documentation](./ProjectDocumentation.md) (this file)
- [Training Notebooks](../notebooks/)
- [Daily Notes](./daily/)

## Notes

_This document will be updated as the project evolves. Use Obsidian's linking features to connect related concepts and create daily logs of progress._
