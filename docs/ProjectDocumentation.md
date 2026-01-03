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
│   └── dqn_baseline.yaml
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

### Phase 3: Simple RL Algorithm 🔄 IN PROGRESS (Jan 2-15, 2026)

- [x] Learn DQN concepts (Q-learning, experience replay, target networks) ✅ 2026-01-02
- [x] Create YAML configuration system for hyperparameters ✅ 2026-01-02
- [x] Create config loader utility (`src/utils/config_loader.py`) ✅ 2026-01-02
- [x] Simplify action space with JoypadSpace (256 → 7 actions) ✅ 2026-01-02
- [x] Build database logging utilities with connection pooling ✅ 2026-01-02
- [x] Create training script structure (main entry point, argument parsing) ✅ 2026-01-02
- [x] Integrate Stable-Baselines3 DQN with configuration ✅ 2026-01-02
- [x] Add custom callbacks for W&B and database logging during training ✅ 2026-01-02
- [x] Test end-to-end training run (short trial to verify everything works) ✅ 2026-01-03
- [ ] Run full DQN training (2M timesteps) 📅 2026-01-03
- [ ] Create evaluation script (load trained model, run test episodes) 📅 2026-01-05
- [ ] Build analysis notebook comparing random vs. trained agent 📅 2026-01-08

**Phase 3 Progress: 9/12 tasks complete (75%)**

**Completed Artifacts:**
- `configs/dqn_baseline.yaml` - Experiment configuration (2M timesteps, CnnPolicy, SIMPLE_MOVEMENT)
- `src/utils/config_loader.py` - YAML configuration loader
- `src/environments/mario_env.py` - Environment helper with simplified actions + CompatibilityWrapper
- `src/environments/wrappers.py` - 5 custom wrappers (Compatibility, Grayscale, Resize, FrameStack, Transpose)
- `src/utils/db_logger.py` - Database logging with connection pooling (5 functions) + metadata tracking
- `src/training/train.py` - Complete training orchestrator with git/version metadata tracking
- `src/training/callbacks.py` - Custom WandbCallback and DatabaseCallback
- Successful 1000-timestep test run ✅

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
- **Systematic debugging**: Fixed 12 integration issues to achieve first successful training run
- **Real ML engineering**: Integration debugging is 50% of the job - tutorials skip this critical skill!

### Phase 4: Advanced Techniques (Weeks 6-9, Feb-Mar 2026)

- [ ] Implement PPO algorithm (often better for platformers) 📅 2026-02-05
- [ ] Experiment with curriculum learning: 📅 2026-02-12
  - [ ] Train on easier levels first 📅 2026-02-09
  - [ ] Gradually increase difficulty 📅 2026-02-12
- [ ] Implement reward shaping: 📅 2026-02-19
  - [ ] Reward for distance traveled 📅 2026-02-15
  - [ ] Penalty for time spent idle 📅 2026-02-17
  - [ ] Bonus for collecting coins/powerups 📅 2026-02-19
- [ ] Add sophisticated preprocessing: 📅 2026-02-26
  - [ ] Attention mechanisms 📅 2026-02-23
  - [ ] State representation learning 📅 2026-02-26
- [ ] Systematic hyperparameter tuning: 📅 2026-03-05
  - [ ] Learning rate schedules 📅 2026-03-01
  - [ ] Network architecture variations 📅 2026-03-03
  - [ ] Exploration/exploitation balance 📅 2026-03-05
- [ ] A/B testing framework for comparing configurations 📅 2026-03-08

### Phase 5: Production & Analysis (Weeks 10-12, Mar 2026)

- [ ] Containerize training environment with Docker: 📅 2026-03-15
  - [ ] Multi-stage build (training vs. inference) 📅 2026-03-12
  - [ ] GPU support configuration 📅 2026-03-15
- [ ] Set up GitHub Actions workflows: 📅 2026-03-22
  - [ ] Run tests on pull requests 📅 2026-03-18
  - [ ] Code quality checks (black, ruff, mypy) 📅 2026-03-19
  - [ ] Automated model evaluation 📅 2026-03-22
- [ ] Create comprehensive data analysis dashboards: 📅 2026-03-29
  - [ ] Training stability analysis 📅 2026-03-25
  - [ ] Hyperparameter correlation studies 📅 2026-03-27
  - [ ] Performance comparison across algorithms 📅 2026-03-29
- [ ] Build model evaluation pipeline: 📅 2026-04-03
  - [ ] Standardized test episodes 📅 2026-03-31
  - [ ] Statistical significance testing 📅 2026-04-02
  - [ ] Performance benchmarking 📅 2026-04-03
- [ ] Write comprehensive documentation: 📅 2026-04-10
  - [ ] API documentation 📅 2026-04-05
  - [ ] Training guides 📅 2026-04-07
  - [ ] Architecture decisions 📅 2026-04-09
  - [ ] Lessons learned 📅 2026-04-10

### Phase 6: Extensions (Ongoing, Apr 2026+)

- [ ] Expand to other games: 📅 2026-04-15
  - [ ] Sonic the Hedgehog 📅 2026-04-15
  - [ ] Contra 📅 2026-04-20
  - [ ] Custom environments 📅 2026-04-25
- [ ] Implement curiosity-driven exploration 📅 2026-05-01
- [ ] Multi-agent training (competitive/cooperative) 📅 2026-05-10
- [ ] Transfer learning between game levels 📅 2026-05-20
- [ ] Model distillation (compress large models) 📅 2026-06-01
- [ ] Real-time inference optimization 📅 2026-06-10
- [ ] Web dashboard for live agent monitoring 📅 2026-06-20

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
