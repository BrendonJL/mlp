# Mario RL Agent - Machine Learning Through Applied Practice

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Learning machine learning by building a reinforcement learning agent that masters Super Mario Bros**

A hands-on journey into deep reinforcement learning - training agents from scratch to play NES Super Mario Bros.

## Training Results

| Agent | Episodes | Avg Reward | Avg Distance | Max Distance | Level Progress |
|-------|----------|------------|--------------|--------------|----------------|
| Random Baseline | - | ~380 | 350 px | 434 px | 11% |
| DQN (2M steps) | 785 | 1,920 | 1,025 px | 2,743 px | 31% |
| PPO v2 (2M) | 2,197 | 866 | 688 px | 2,226 px | 21% |
| PPO v3 (10M) | 4,684 | 949 | 859 px | 2,226 px | 26% |
| **PPO v4 (10M + skip)** | 18,958 | 4,448 | 1,949 px | 3,154 px | **60%** |
| PPO v5 (5M) | 7,430 | 1,123 | 1,399 px | 3,157 px | Failed* |
| PPO v6 | - | - | - | - | Failed* |
| **PPO v7 (10M + RAM)** | 16,571 | **6,151** | **2,321 px** | 2,776 px | **71%** |

*v5 failed due to reward engineering issues (negative reward/distance correlation). v6 failed due to entropy collapse.

**Key Insights:**
- **Frame skip (4 frames/action)** was the v4 breakthrough - reduced jump chaining difficulty by 4x
- **RAM observations** (v7) provide more consistent performance - 72% of episodes reach 2700+ vs 42% for v4
- **v7 completed level 1** during evaluation (died on level 2) - the final trained model can win!

## Phase 5 Complete!

Two successful approaches discovered:
- **v4 (CNN + Frame Skip):** Higher variance, occasionally reaches 3154px (near flagpole)
- **v7 (RAM + MLP):** More consistent, 72% of episodes reach 2700+, completed level in evaluation

**Next:** Phase 6 - Imitation Learning (BC pre-training on speedrun data)

## Project Structure

```
mlp/
├── src/                  # Source code
│   ├── environments/    # Game wrappers (SkipFrame, RewardShaping, RAM obs)
│   ├── training/        # Training loops and callbacks
│   └── utils/           # Config loading, database logging
├── configs/             # Hyperparameter configs (v2-v7)
├── models/              # Saved model checkpoints
├── notebooks/           # Jupyter analysis notebooks
├── scripts/             # Evaluation scripts
├── database/            # SQL schemas and migrations
└── docs/                # Documentation and daily logs
```

## Tech Stack

| Category | Tools |
|----------|-------|
| **Core ML** | PyTorch, Stable-Baselines3, Gymnasium, gym-super-mario-bros |
| **Tracking** | PostgreSQL, Weights & Biases |
| **Development** | Poetry, pre-commit hooks |

## Project Journey

| Phase | Status | Key Achievement |
|-------|--------|-----------------|
| 1. Environment Setup | Complete | PostgreSQL + W&B infrastructure |
| 2. Baseline Agent | Complete | Random baseline with 13 metrics |
| 3. DQN Training | Complete | 5.3x improvement over random |
| 4. PPO Implementation | Complete | Learned from policy collapse |
| 5. Tuning & Optimization | **Complete** | v4 (60% progress) + v7 (71% progress, level complete!) |
| 6. Imitation Learning | **Next** | Combine v4+v7 strengths + speedrun pre-training |

## Documentation

- **[Project Architecture](docs/ProjectDocumentation.md)** - Full details and phase documentation
- **[Daily Learning Log](docs/daily/)** - Day-by-day progress
- **[Analysis Notebooks](notebooks/)** - Interactive analysis

## Long-term Goal

Apply these ML techniques to cybersecurity: Suricata rule generation, intelligent incident reporting, and threat detection.

---

*MIT License*
