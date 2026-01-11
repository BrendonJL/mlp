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

### Phase 5: Infrastructure Fixes, Reward Shaping & Hyperparameter Tuning (Jan 2026)

**Part A: Fix Infrastructure (Prerequisites)**
- [ ] Fix callbacks for vectorized environments:
  - [ ] Update `WandbCallback` to iterate over all `n_envs`
  - [ ] Update `DatabaseCallback` to iterate over all `n_envs`
  - [ ] Test callbacks with short PPO run to verify logging works
- [ ] Reduce parallel environments (8 → 4) to prevent CPU throttling
- [ ] Verify episode metrics appear in W&B and PostgreSQL

**Part B: Hyperparameter Tuning**
- [ ] Lower learning rate: 0.0001 → 0.00003
- [ ] Increase entropy coefficient: 0.01 → 0.02
- [ ] Consider early stopping based on evaluation performance
- [ ] Learning rate schedules
- [ ] Batch size and n_steps optimization

**Part C: Reward Shaping**
- [ ] Implement custom reward wrapper:
  - [ ] Bonus for distance traveled (encourage forward progress)
  - [ ] Penalty for time spent idle (discourage standing still)
  - [ ] Reward for collecting coins/powerups
  - [ ] Penalty for losing lives

**Part D: Advanced Techniques**
- [ ] Experiment with curriculum learning:
  - [ ] Train on easier levels first
  - [ ] Gradually increase difficulty
- [ ] A/B testing framework for comparing configurations
- [ ] Goal: Achieve level completion (reach the flag!)

### Phase 6: Imitation Learning (Feb-Mar 2026)

- [ ] Research imitation learning approaches:
  - [ ] Behavioral Cloning (BC)
  - [ ] DAgger (Dataset Aggregation)
  - [ ] GAIL (Generative Adversarial Imitation Learning)
- [ ] Collect expert demonstrations:
  - [ ] Manual gameplay recording
  - [ ] State-action pair extraction
- [ ] Implement imitation learning pipeline:
  - [ ] Pre-train policy from demonstrations
  - [ ] Fine-tune with RL (hybrid approach)
- [ ] Compare pure RL vs imitation-assisted learning
- [ ] Document effectiveness of human demonstrations

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

### Phase 8: Extensions (Ongoing, Apr 2026+)

- [ ] Expand to other games:
  - [ ] Sonic the Hedgehog
  - [ ] Contra
  - [ ] Custom environments
- [ ] Implement curiosity-driven exploration
- [ ] Multi-agent training (competitive/cooperative)
- [ ] Transfer learning between game levels
- [ ] Model distillation (compress large models)
- [ ] Real-time inference optimization
- [ ] Web dashboard for live agent monitoring

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
