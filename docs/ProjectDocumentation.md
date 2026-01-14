---
id: ProjectDocumentation
aliases: []
tags: []
---

id: ProjectDocumentation
aliases: []
tags: []

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
│   └── ppo_baseline.yaml
├── data/                 # Training logs, gameplay videos, episode data
│   ├── logs/
│   └── videos/
├── database/             # SQL schemas, migration scripts for experiment metadata
│   ├── schema.sql
│   └── schema_migration_01.sql
├── docker/               # Dockerfiles for containerized training/deployment (planned)
├── docs/                 # Project documentation and notes (Obsidian vault)
│   ├── daily/           # Daily logs and progress notes
│   ├── templates/       # Note templates
│   ├── ProjectDocumentation.md
│   └── Tasks Dashboard.md
├── .github/
│   └── workflows/       # CI/CD pipelines for automated testing (planned)
├── models/               # Saved model checkpoints and weights
│   ├── checkpoints/
│   └── dqn_baseline_world1-1_final.zip
├── notebooks/            # Jupyter notebooks for analysis and exploration
│   └── 01_environment_exploration.ipynb
├── scripts/              # Utility scripts for testing and exploration
│   ├── random_agent.py
│   └── test_explore_env.py
├── src/                  # Source code
│   ├── agents/          # RL agent implementations (DQN, PPO) (planned)
│   ├── environments/    # Gym environment wrappers and preprocessing
│   │   ├── mario_env.py
│   │   ├── vec_mario_env.py
│   │   └── wrappers.py
│   ├── models/          # Neural network architectures (planned)
│   ├── preprocessing/   # Frame processing utilities (planned)
│   ├── training/        # Training loops and callbacks
│   │   ├── callbacks.py
│   │   └── train.py
│   ├── utils/           # Helper functions and utilities
│   │   ├── config_loader.py
│   │   └── db_logger.py
│   └── __init__.py
├── tests/               # Unit tests for components (planned)
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

### Phase 5: Infrastructure Fixes, Reward Shaping & Hyperparameter Tuning ⏳ IN PROGRESS (Jan 11-13, 2026)

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
- [x] Clean up PostgreSQL database ✅ 2026-01-12
- [x] Generate visualizations ✅ 2026-01-12
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

**Part E: Frame Skip Optimization** ⏳ PENDING
- [ ] Implement `SkipFrame` wrapper (skip=4 frames per action)
- [ ] Add wrapper to environment pipeline (early in chain, before preprocessing)
- [ ] Create `configs/ppo_v4.yaml` with frame skip enabled
- [ ] Run 2M step test with frame skip
- [ ] Compare results: Does frame skip break past the 722 barrier?
- [ ] Document findings and decide on final config for Phase 6

**Why Frame Skip Matters:**
Without frame skip, the agent processes every frame at 60 FPS:
- To "hold jump" for 0.5 seconds = must output "jump" 30 consecutive times
- This is the core of the "Tall Pipe Problem"

With frame skip (skip=4):
- Each action repeats for 4 frames automatically
- To "hold jump" for 0.5 seconds = only need ~7 consecutive "jump" actions
- Dramatically reduces exploration difficulty

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
- `src/environments/wrappers.py` - `RewardShapingWrapper` with milestone bonuses
- `src/environments/vec_mario_env.py` - Added `VecMonitor` wrapping
- `src/training/callbacks.py` - Fixed vectorized environment iteration
- `src/training/train.py` - Added `linear_schedule()` function for LR annealing
- `notebooks/03_ppo_vs_dqn_comparison.ipynb` - Full analysis notebook

**Key Learnings:**
- SubprocVecEnv runs environments in separate processes - print statements don't appear in main terminal
- Reward shaping alone doesn't make episodes end faster - need early termination too
- On-policy algorithms (PPO) need stronger reward signals than off-policy (DQN) because they discard data after each update
- Gym vs Gymnasium API differences: old gym returns 4 values from step(), new gymnasium returns 5
- **Deterministic vs Stochastic evaluation**: `deterministic=False` samples from policy (matches training), `deterministic=True` always picks highest probability
- **Action space limitations**: SIMPLE_MOVEMENT (7 actions) can't "hold" buttons - high jumps require chaining consecutive jump actions
- **Training duration matters**: Research showed successful Mario PPO runs used 10M+ steps with LR scheduling

### Phase 6: Imitation Learning ⏳ NEXT (Jan 2026)

**Goal:** Teach the agent speedrunning techniques it can't discover through random exploration.

**Planned Approach: BC Pre-training → PPO Fine-tuning**

This hybrid approach leverages both demonstration data and reinforcement learning:
1. **Behavioral Cloning (BC)**: Supervised learning on expert (state, action) pairs to initialize policy
2. **PPO Fine-tuning**: Continue training with RL to optimize beyond demonstrations
3. **Optional reward shaping**: Bonus reward when agent's action matches expert

**Data Sources (Two-Stage Plan):**

| Stage | Source | Purpose | Format Conversion |
|-------|--------|---------|-------------------|
| Stage 1 | [rafaelcp/smbdataset](https://github.com/rafaelcp/smbdataset) | Competent play (737k frames, 280 episodes) | 256x240 → 84x84, 256 actions → 7 actions |
| Stage 2 | TAS input files from [TASVideos.org](https://tasvideos.org) | Speedrun tricks & glitches | Parse .fm2 files, replay through environment |

**Why Two Stages:**
- **Stage 1 (smbdataset)**: General competent play - clearing obstacles, basic strategies
- **Stage 2 (TAS)**: Specific speedrun tricks that require precise inputs (wall clips, momentum glitches)

**Action Space Considerations:**
- Current: `SIMPLE_MOVEMENT` (7 actions) - no left+jump combinations
- Speedrun tricks may require `COMPLEX_MOVEMENT` (12 actions) or custom action space
- More actions = longer training (exploration scales combinatorially, not linearly)
- Custom action space option: Add only `left+A` for specific tricks without full complexity

**Reward Shaping Adjustments:**
- Current backward penalty (-0.1/pixel left) discourages learning tricks requiring leftward movement
- For speedrun training: reduce penalty to -0.05 or make context-dependent
- Let imitation learning handle the 5% exception cases where left movement is optimal

**Technical Approach:**
- **Frame skip (skip=4)** included in environment pipeline (validated in Phase 5 Part E)
- Transfer CNN feature layers from trained PPO model (visual understanding transfers)
- Retrain output layers for new action space if switching to COMPLEX_MOVEMENT
- Use demonstrations to guide exploration past obstacles agent is stuck on

**Tasks:**
- [ ] Download and preprocess smbdataset (format conversion)
- [ ] Implement BC training pipeline (supervised learning on demos)
- [ ] Test BC pre-training → PPO fine-tuning workflow
- [ ] Research TAS file format (.fm2) and conversion to action indices
- [ ] Create demo replay script to capture (observation, action) pairs from TAS
- [ ] Evaluate custom action space (SIMPLE + left+A) vs full COMPLEX_MOVEMENT
- [ ] Adjust reward shaping for speedrun-friendly training
- [ ] Train and evaluate imitation-augmented agent
- [ ] Compare: Pure RL vs BC-initialized vs Full imitation pipeline

**Key Concepts to Learn:**
- Behavioral Cloning: Supervised learning on expert demonstrations
- Credit assignment problem: Connecting immediate actions to delayed rewards
- Policy distillation: Using one trained agent's outputs to teach another
- On-policy vs off-policy implications for demonstration data

### Phase 7: Production & Analysis (Mar-Apr 2026)

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
