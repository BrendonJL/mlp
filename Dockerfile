# ── Base image ──────────────────────────────────────────────
# Slim Debian with Python 3.13, ~150MB starting point
FROM python:3.13-slim AS base

# ── System dependencies ─────────────────────────────────────
# nes_py (NES emulator) needs C build tools and display libs
# We install, build, then clean up to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ───────────────────────────────────────
WORKDIR /app

# ── Install Poetry ──────────────────────────────────────────
# pipx-style isolated install so Poetry doesn't pollute project deps
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

# ── Copy dependency files first ─────────────────────────────
# Docker caches layers — if these files don't change, deps
# won't reinstall on every code change (huge time saver)
COPY pyproject.toml poetry.lock* ./

# ── Install project dependencies (no dev tools) ────────────
RUN poetry install --only main --no-interaction --no-ansi

# ── Copy source code ───────────────────────────────────────
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY scripts/ scripts/

# ── TODO(human): Write the entrypoint ──────────────────────
# This is the default command when someone runs the container.
# It should run training using a config file.
# Hint: look at how you start training in the distrobox.
# The syntax is: CMD ["python", "-m", "module.path", "--flag", "value"]
CMD ["python", "-m", "src.training.train", "--config", "configs/ppo_v9.yaml"]
