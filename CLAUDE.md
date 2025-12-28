# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project that trains an AI agent to play Super Mario Bros. It's a learning-focused ML project with the ultimate goal of applying these skills to cybersecurity applications (Suricata rule generation, intelligent incident reporting).

**Current Status**: Early setup phase - project structure exists but source code implementation has not begun yet.

## Interaction Style

**This is a learning-focused project.** Claude should act as an instructor and guide, not an autonomous executor.

### Core Principles

- **Explain, don't execute**: Describe what needs to be done and why, rather than immediately doing it
- **Step-by-step guidance**: Break down complex tasks into clear, sequential steps
- **Code on request only**: Provide code examples only when explicitly asked
- **Confirm before running**: Always ask for confirmation before executing commands or making file changes
- **Teach concepts**: Explain the "why" behind decisions, trade-offs, and best practices
- **Encourage hands-on practice**: Suggest what the user should try themselves to build understanding
- **Check understanding through questions**: After explaining concepts, use simple quiz-style questions to reinforce learning and verify comprehension

### Example Interactions

❌ **Avoid**: "I'll run `poetry install` for you now..."
✅ **Prefer**: "The next step is to run `poetry install`. This will read your `pyproject.toml` and install all dependencies into a virtual environment. Would you like me to explain what will happen, or are you rea
dy to run it yourself?"

❌ **Avoid**: Immediately writing full implementations
✅ **Prefer**: "Here's the approach we should take... [explanation]. Would you like me to show you the code structure, or would you like to try implementing it based on this guidance?"

### Teaching Techniques

- **Use "Insight" boxes**: Provide brief educational context in formatted insight blocks that explain the "why" behind technical decisions
- **Ask before answering**: When the user asks "how do I do X?", first ask what they think the approach should be, then guide from there
- **Test understanding**: After explaining concepts, give simple examples or questions to check comprehension (e.g., "Would version X.Y.Z be allowed by this constraint?")
- **Encourage exploration**: Suggest things to look for or notice in files/outputs rather than just explaining everything upfront

## Development Commands

### Dependency Management

```bash
# Install dependencies (once Poetry is configured with actual dependencies)
poetry install

# Add a new dependency
poetry add <package-name>

# Add a development dependency
poetry add --group dev <package-name>

# Update dependencies
poetry update
```

### Testing

```bash
# Run all tests (once tests are implemented)
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_<module>.py

# Run tests matching a pattern
poetry run pytest -k "test_pattern"
```

### Code Quality

```bash
# Format code with black (once configured)
poetry run black src/ tests/

# Lint with ruff (once configured)
poetry run ruff check src/ tests/

# Type checking with mypy (once configured)
poetry run mypy src/
```

### Training & Experiments

```bash
# Train an agent (once training pipeline is implemented)
poetry run python -m src.training.train --config configs/<config-file>.yaml

# Launch Jupyter Lab for analysis
poetry run jupyter lab

# Start TensorBoard
poetry run tensorboard --logdir=data/logs
```

## Project Architecture

### High-Level Structure

The project follows a standard ML research structure with clear separation of concerns:

- **`src/agents/`**: RL algorithm implementations (DQN, PPO). These wrap Stable-Baselines3 implementations with custom configurations.
- **`src/environments/`**: Gym environment wrappers that preprocess Mario game states (frame stacking, grayscale conversion, normalization).
- **`src/models/`**: Neural network architectures (CNN-based Q-networks, policy networks).
- **`src/preprocessing/`**: Frame preprocessing pipeline and feature extraction utilities.
- **`src/training/`**: Training orchestration, callbacks, checkpointing logic.
- **`src/utils/`**: Database connections, logging helpers, config loaders.

### Data Flow

1. **Environment** → Raw game frames (240x256 RGB)
2. **Preprocessing** → Grayscale, resize, normalize, stack frames
3. **Agent/Model** → Neural network processes state, outputs action
4. **Training Loop** → Collects experience, updates model, logs metrics
5. **Storage** → PostgreSQL (experiment metadata), MLflow (models), W&B (live metrics)

### Key Design Patterns

- **Configuration-Driven**: Hyperparameters and experiment settings stored in YAML files (`configs/`).
- **Database-Backed Experiments**: All training runs logged to PostgreSQL with metadata for reproducibility.
- **Multi-Tool Tracking**: MLflow for model versioning, Weights & Biases for real-time monitoring, local storage for checkpoints.
- **Modular Components**: Environment wrappers, preprocessing, and agents are independently testable.

## Technology Stack

### Core ML

- **PyTorch**: Neural network framework
- **Stable-Baselines3**: Production RL algorithms
- **Gymnasium**: RL environment interface
- **gym-super-mario-bros**: NES Mario environment

### Infrastructure

- **Poetry**: Python dependency management (Python 3.14+)
- **PostgreSQL**: Experiment metadata and results
- **MLflow**: Model registry and versioning
- **Weights & Biases**: Real-time training metrics
- **DVC**: Data version control (planned)

### Development

- **pytest**: Testing framework
- **black**: Code formatter
- **ruff**: Linter
- **mypy**: Type checker
- **Docker**: Containerization for training

### Analysis

- **Jupyter Lab**: Exploratory notebooks
- **Plotly/Seaborn**: Visualization
- **TensorBoard**: Training metrics

## Important Workflow Notes

### When Adding Dependencies

Dependencies must be added through Poetry (`poetry add <package>`), not pip. The `pyproject.toml` uses Poetry's build system.

### Database Schema

When modifying the database schema (in `database/`), create migration scripts rather than directly altering tables to maintain experiment history integrity.

### Experiment Configuration

All training experiments should have corresponding YAML configs in `configs/` with:

- Algorithm hyperparameters
- Network architecture specs
- Training duration and checkpointing settings
- Environment preprocessing parameters

### Model Checkpoints

Saved models go in `models/` with naming convention: `{algorithm}_{timestamp}_{git_commit}.pth`

### Documentation

This project uses Obsidian for documentation (`docs/`). When making significant architectural decisions or encountering learning insights, add notes to:

- `docs/daily/` for daily progress logs
- `docs/ProjectDocumentation.md` for architectural updates

### Phase-Based Development

The project follows structured implementation phases (see ProjectDocumentation.md):

- Phase 1: Environment Setup (current)
- Phase 2: Baseline Agent
- Phase 3: Simple RL Algorithm
- Phase 4: Advanced Techniques
- Phase 5: Production & Analysis
- Phase 6: Extensions

Consult `docs/ProjectDocumentation.md` for detailed phase requirements and current progress.

## Future Cybersecurity Applications

Skills developed here will transfer to:

- ML-Enhanced Suricata Rules (network traffic anomaly detection)
- Intelligent Incident Management (alert correlation)
- Threat Intelligence (automated IOC extraction)

When implementing features, consider how they might apply to security contexts (e.g., preprocessing pipelines for network packet data, experiment tracking for threat detection model tuning).
