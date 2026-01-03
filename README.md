# 🎮 Mario RL Agent - Machine Learning Through Applied Practice

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Learning machine learning by building a reinforcement learning agent that masters Super Mario Bros**

## 🎯 Project Overview

This project represents my hands-on journey into machine learning through practical application. Rather than starting with pure theory, I'm building a deep reinforcement learning agent capable of learning to play Super Mario Bros from scratch - demonstrating fundamental ML concepts including neural networks, training pipelines, experiment tracking, and model evaluation.

**Why Mario?** It's the perfect learning environment: complex enough to be challenging, simple enough to understand, and engaging enough to stay motivated. Plus, the skills transfer directly to real-world applications.

**Long-term Goal:** Apply these ML techniques to cybersecurity challenges, specifically Suricata rule generation and intelligent incident reporting systems.

## 🚀 Key Features

- **Deep Reinforcement Learning**: Implementation of DQN and PPO algorithms
- **Experiment Tracking**: MLflow and Weights & Biases integration for reproducible research
- **Production-Ready Pipeline**: Dockerized training environment with CI/CD
- **Comprehensive Documentation**: Detailed notes on architecture decisions and learning process
- **Data-Driven Analysis**: PostgreSQL storage with Jupyter notebook visualizations

## 📁 Project Structure

```
mlp/
├── src/                  # Source code
│   ├── agents/          # RL agent implementations (planned)
│   ├── environments/    # Game environment wrappers + preprocessing
│   │   ├── mario_env.py
│   │   └── wrappers.py
│   ├── models/          # Neural network architectures (planned)
│   ├── training/        # Training loops and callbacks
│   │   ├── train.py
│   │   └── callbacks.py
│   └── utils/           # Helper functions
│       ├── config_loader.py
│       └── db_logger.py
├── configs/             # Hyperparameter configurations (YAML)
│   └── dqn_baseline.yaml
├── models/              # Saved model checkpoints
│   └── dqn_baseline_world1-1_final.zip
├── notebooks/           # Jupyter analysis notebooks
│   └── 01_environment_exploration.ipynb
├── scripts/             # Testing and exploration scripts
│   ├── random_agent.py
│   └── test_explore_env.py
├── database/            # SQL schemas and migrations
│   ├── schema.sql
│   └── schema_migration_01.sql
├── docs/                # Project documentation (Obsidian vault)
│   ├── ProjectDocumentation.md
│   └── daily/          # Learning journal
├── tests/              # Unit tests (planned)
├── docker/             # Container configurations (planned)
├── CLAUDE.md           # Instructions for Claude Code
└── .pre-commit-config.yaml  # Code quality automation
```

## 🛠️ Tech Stack

**Core ML**

- PyTorch - Deep learning framework
- Stable-Baselines3 - RL algorithms
- Gymnasium - Environment interface
- gym-super-mario-bros - NES Mario environment

**Data & Infrastructure**

- PostgreSQL - Experiment metadata
- MLflow - Model versioning
- Weights & Biases - Real-time metrics
- DVC - Data version control

**Development**

- Poetry - Dependency management
- Docker - Containerization
- GitHub Actions - CI/CD
- pytest - Testing framework

## 📚 Documentation

- **[Project Architecture](docs/ProjectDocumentation.md)** - Comprehensive project overview, tech stack, and implementation phases
- **[Daily Learning Log](docs/daily/)** - Day-by-day progress and insights
- **[GitHub Repository](https://github.com/BrendonJL/mlp)** - Source code and version history

## 🎓 Learning Objectives

- [x] Set up production-grade ML project structure
- [x] Configure PostgreSQL for experiment tracking
- [x] Implement database schema design with relational integrity
- [x] Establish baseline metrics for RL environments
- [x] Build preprocessing pipelines for game state observations
- [x] Integrate cloud experiment tracking (Weights & Biases)
- [ ] Master reinforcement learning fundamentals (DQN, PPO)
- [ ] Build and train custom neural network architectures
- [ ] Implement reproducible experiment configurations (YAML)
- [ ] Deploy containerized ML applications
- [ ] Apply ML to real-world security problems

## 🚧 Current Status

**✅ Phase 1: Environment Setup** (Complete - Dec 26-29, 2025)

The foundation is solid! Completed in 3 days:
- ✅ Project structure with Poetry dependency management
- ✅ Git workflow and GitHub integration
- ✅ Obsidian documentation system with daily logs
- ✅ PostgreSQL database with 4-table schema design
- ✅ Weights & Biases cloud experiment tracking
- ✅ Pre-commit hooks for automated code quality

**✅ Phase 2: Baseline Agent** (Complete - Dec 30-31, 2025)

Production-quality baseline established:
- ✅ Random agent implementation with 10-episode baseline run
- ✅ Enhanced metrics tracking (13 comprehensive values)
- ✅ Frame preprocessing pipeline (grayscale, resize, normalize, frame stack)
- ✅ Weights & Biases cloud logging with authentication
- ✅ PostgreSQL schema migration for episode metrics
- ✅ Jupyter notebook for environment exploration
- ✅ Success criteria defined for Phase 3 (x_pos > 434, score ≥ 100, flag_get = True)

**Baseline Performance:**
- Average reward: ~380
- Max distance (x_pos): 434
- Level completions: 0/10 (expected for random)
- Episode length: 1000 steps (always timeout)

**🔄 Phase 3: Simple RL Algorithm** (In Progress - Jan 2-15, 2026)

Progress: 9/12 tasks complete (75%)

✅ Completed:
- DQN conceptual learning (Q-networks, experience replay, target networks)
- YAML configuration system for reproducible experiments
- Complete training pipeline with git metadata tracking
- Custom Gym wrappers (5 total: Compatibility, Grayscale, Resize, FrameStack, Transpose)
- Database logging utilities with connection pooling
- Wandb and PostgreSQL integration callbacks
- **First successful end-to-end training run (1000 timesteps)!** ✅

⏳ Next:
- Run full 2M timestep DQN training
- Create evaluation script
- Build analysis notebook

See [ProjectDocumentation.md](docs/ProjectDocumentation.md) for complete timeline and detailed implementation phases.

### Recent Highlights

**Jan 3, 2026** - **HUGE MILESTONE!** After 3 hours of systematic debugging, achieved first successful end-to-end training run! Fixed 12 integration issues including module imports, config mismatches, API compatibility between old Gym and new Gymnasium, image format for PyTorch, and wrapper design. Built 5 custom wrappers following Single Responsibility Principle. Added git commit hash, Python version, and PyTorch version tracking for reproducibility. Database and wandb logging verified. **Ready for full 2M timestep training!** 🚀

**Dec 31, 2025** - Phase 2 complete! Built comprehensive random baseline with 13 tracked metrics (x_pos, score, time, coins, life, status, flag_get, etc.), integrated wandb cloud tracking with authentication, extended database schema with 8 new episode metric columns, and established success criteria for Phase 3. Attempted video recording but encountered gym-super-mario-bros render limitations - pragmatically chose metrics-only approach. **Ready for DQN training!** 🚀

**Dec 29, 2025** - Designed and implemented complete database schema for ML experiment tracking. Learned SQL CREATE TABLE syntax, foreign key relationships, and the Entity-Attribute-Value pattern for flexible hyperparameter storage. Set up W&B and pre-commit hooks. Phase 1 complete! 🎉

## 🎯 Future Applications

The skills developed here will transfer to:

- **ML-Enhanced Suricata Rules** - Anomaly detection in network traffic
- **Intelligent Incident Management** - Alert correlation and prioritization
- **Threat Intelligence** - Automated IOC extraction and classification

## 📝 License

MIT License - feel free to learn from this project!

## 🤝 Connect

This is a learning project - feedback and suggestions welcome!

---

_"The best way to learn machine learning is by building something real."_
