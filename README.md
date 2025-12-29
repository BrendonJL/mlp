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
│   ├── agents/          # RL agent implementations (DQN, PPO)
│   ├── environments/    # Game environment wrappers
│   ├── models/          # Neural network architectures
│   ├── training/        # Training loops and callbacks
│   └── utils/           # Helper functions
├── configs/             # Hyperparameter configurations
├── notebooks/           # Jupyter analysis notebooks
├── docs/                # Project documentation
│   ├── ProjectDocumentation.md
│   └── daily/          # Learning journal
├── tests/              # Unit tests
└── docker/             # Container configurations
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
- [ ] Master reinforcement learning fundamentals
- [ ] Build and train custom neural network architectures
- [ ] Implement experiment tracking and reproducibility
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

**🎯 Next Up: Phase 2 - Baseline Agent** (Starting Jan 2026)

- Install gym-super-mario-bros environment
- Implement random agent for baseline metrics
- Build frame preprocessing pipeline
- Create first Jupyter analysis notebook

See [ProjectDocumentation.md](docs/ProjectDocumentation.md) for complete timeline and detailed implementation phases.

### Recent Highlights

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
