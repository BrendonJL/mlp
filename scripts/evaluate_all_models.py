"""
evaluate_all_models.py

Evaluate all trained models from the MLP project history in a single run,
including a random baseline. Produces a comparison table and saves a
markdown report.

Usage:
    python scripts/evaluate_all_models.py
    python scripts/evaluate_all_models.py --num-episodes 10 --models random,ppo_v7,ppo_v8
    python scripts/evaluate_all_models.py --output docs/my_eval.md
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Resolve project root so imports work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT  # noqa: E402

from src.environments.mario_env import make_mario_env  # noqa: E402

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    key: str
    path: str | None  # None for random baseline
    algorithm: str  # "DQN", "PPO", or "random"
    action_space: list  # gym action list
    obs_mode: str  # "pixel" or "ram"
    reward_wrapper: str  # "standard" or "speedrun"
    frame_skip: int = 4


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        key="random",
        path=None,
        algorithm="random",
        action_space=SIMPLE_MOVEMENT,
        obs_mode="pixel",
        reward_wrapper="standard",
    ),
    ModelSpec(
        key="dqn_baseline",
        path="models/dqn_baseline_world1-1_final.zip",
        algorithm="DQN",
        action_space=SIMPLE_MOVEMENT,
        obs_mode="pixel",
        reward_wrapper="standard",
    ),
    ModelSpec(
        key="ppo_baseline",
        path="models/ppo_baseline_world1-1_final.zip",
        algorithm="PPO",
        action_space=SIMPLE_MOVEMENT,
        obs_mode="pixel",
        reward_wrapper="standard",
    ),
    ModelSpec(
        key="ppo_v2",
        path="models/ppo_v2_world1-1_final.zip",
        algorithm="PPO",
        action_space=SIMPLE_MOVEMENT,
        obs_mode="pixel",
        reward_wrapper="standard",
    ),
    ModelSpec(
        key="ppo_v3",
        path="models/ppo_v3_world1-1_recovered.zip",
        algorithm="PPO",
        action_space=SIMPLE_MOVEMENT,
        obs_mode="pixel",
        reward_wrapper="standard",
    ),
    ModelSpec(
        key="ppo_v4",
        path="models/ppo_v4_world1-1_recovered.zip",
        algorithm="PPO",
        action_space=SIMPLE_MOVEMENT,
        obs_mode="pixel",
        reward_wrapper="standard",
    ),
    ModelSpec(
        key="ppo_v5",
        path="models/ppo_v5_world1-1_recovered.zip",
        algorithm="PPO",
        action_space=RIGHT_ONLY,
        obs_mode="pixel",
        reward_wrapper="speedrun",
    ),
    ModelSpec(
        key="ppo_v7",
        path="models/ppo_v7_world1-1_recovered.zip",
        algorithm="PPO",
        action_space=SIMPLE_MOVEMENT,
        obs_mode="ram",
        reward_wrapper="standard",
    ),
    ModelSpec(
        key="ppo_v8",
        path="models/ppo_v8_world1-1_final.zip",
        algorithm="PPO",
        action_space=SIMPLE_MOVEMENT,
        obs_mode="ram",
        reward_wrapper="standard",
    ),
    ModelSpec(
        key="ppo_v9",
        path="models/ppo_v9_world1-1_final.zip",
        algorithm="PPO",
        action_space=SIMPLE_MOVEMENT,
        obs_mode="ram",
        reward_wrapper="standard",
    ),
]

REGISTRY_BY_KEY: dict[str, ModelSpec] = {m.key: m for m in MODEL_REGISTRY}

# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    reward: float
    x_pos: int  # max x_pos reached during episode
    score: int
    coins: int
    steps: int
    flag_get: bool
    wall_start: float = 0.0  # time.time() when episode started
    wall_end: float = 0.0  # time.time() when episode ended


@dataclass
class ModelResult:
    key: str
    episodes: list[EpisodeResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    # Computed summary stats (populated after all episodes)
    avg_reward: float = 0.0
    avg_distance: float = 0.0
    max_distance: int = 0
    completion_rate: float = 0.0
    avg_steps: float = 0.0
    avg_score: float = 0.0
    avg_coins: float = 0.0

    def compute_stats(self) -> None:
        if not self.episodes:
            return
        self.avg_reward = float(np.mean([e.reward for e in self.episodes]))
        self.avg_distance = float(np.mean([e.x_pos for e in self.episodes]))
        self.max_distance = int(np.max([e.x_pos for e in self.episodes]))
        completions = sum(1 for e in self.episodes if e.flag_get or e.x_pos >= 3150)
        self.completion_rate = completions / len(self.episodes) * 100.0
        self.avg_steps = float(np.mean([e.steps for e in self.episodes]))
        self.avg_score = float(np.mean([e.score for e in self.episodes]))
        self.avg_coins = float(np.mean([e.coins for e in self.episodes]))


# ---------------------------------------------------------------------------
# Environment & model helpers
# ---------------------------------------------------------------------------


def load_model(spec: ModelSpec, env: Any) -> Any:
    """Load SB3 model. Returns None for random baseline."""
    if spec.algorithm == "random":
        return None

    model_path = PROJECT_ROOT / spec.path  # type: ignore[arg-type]

    if spec.algorithm == "DQN":
        from stable_baselines3 import DQN

        return DQN.load(str(model_path), env=env)
    elif spec.algorithm == "PPO":
        from stable_baselines3 import PPO

        return PPO.load(str(model_path), env=env)
    else:
        raise ValueError(f"Unknown algorithm: {spec.algorithm}")


def run_episode(
    env: Any, model: Any, max_steps: int = 5000, timeout: float | None = None
) -> EpisodeResult:
    """
    Run a single episode. Returns metrics collected during the episode.
    Completion is determined by flag_get OR x_pos >= 3150 at any point.
    """
    obs, info = env.reset()
    wall_start = time.time()

    total_reward = 0.0
    max_x_pos = info.get("x_pos", 0)
    last_score = info.get("score", 0)
    last_coins = info.get("coins", 0)
    flag_get = False
    steps = 0

    for _ in range(max_steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if timeout and (time.time() - wall_start) > timeout:
            break

        current_x = info.get("x_pos", 0)
        if current_x > max_x_pos:
            max_x_pos = current_x

        last_score = info.get("score", 0)
        last_coins = info.get("coins", 0)

        if info.get("flag_get", False):
            flag_get = True

        if terminated or truncated:
            break

    wall_end = time.time()

    return EpisodeResult(
        reward=total_reward,
        x_pos=max_x_pos,
        score=last_score,
        coins=last_coins,
        steps=steps,
        flag_get=flag_get,
        wall_start=wall_start,
        wall_end=wall_end,
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_model(
    spec: ModelSpec, num_episodes: int, render: bool = False
) -> ModelResult:
    result = ModelResult(key=spec.key)

    # Check model file exists (skip missing zips)
    if spec.path is not None:
        model_path = PROJECT_ROOT / spec.path
        if not model_path.exists():
            result.skipped = True
            result.skip_reason = f"File not found: {model_path}"
            print(f"  [SKIP] {spec.key}: {result.skip_reason}")
            return result

    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {spec.key}  ({spec.algorithm})")
    print(
        f"  Action space: {'RIGHT_ONLY' if spec.action_space is RIGHT_ONLY else 'SIMPLE_MOVEMENT'}"
    )
    print(f"  Obs mode:     {spec.obs_mode}")
    print(f"  Reward:       {spec.reward_wrapper}")
    print(f"  Render:       {'human' if render else 'headless'}")
    print(f"{'=' * 60}")

    # Build environment matching training config
    env = make_mario_env(
        game_version="SuperMarioBros-v3",
        action_space=spec.action_space,
        render_mode="human" if render else None,
        skip=spec.frame_skip,
        reward_wrapper=spec.reward_wrapper,
        observation_mode=spec.obs_mode,
    )

    try:
        model = load_model(spec, env)

        for ep in range(1, num_episodes + 1):
            ep_result = run_episode(
                env, model, max_steps=5000, timeout=30 if render else None
            )
            result.episodes.append(ep_result)

            completed_marker = (
                " [FLAG]" if ep_result.flag_get or ep_result.x_pos >= 3150 else ""
            )
            ts = datetime.fromtimestamp(ep_result.wall_start).strftime("%H:%M:%S")
            print(
                f"  ep {ep:3d}/{num_episodes}  "
                f"reward={ep_result.reward:8.1f}  "
                f"x_pos={ep_result.x_pos:4d}  "
                f"score={ep_result.score:5d}  "
                f"steps={ep_result.steps:4d}  "
                f"@{ts}"
                f"{completed_marker}"
            )
    finally:
        env.close()

    result.compute_stats()

    # Log best episode timestamp for video trimming
    if result.episodes:
        best_ep = max(result.episodes, key=lambda e: e.x_pos)
        best_idx = result.episodes.index(best_ep) + 1
        best_start = datetime.fromtimestamp(best_ep.wall_start).strftime("%H:%M:%S")
        best_end = datetime.fromtimestamp(best_ep.wall_end).strftime("%H:%M:%S")
        duration = best_ep.wall_end - best_ep.wall_start
        print(
            f"\n  >> BEST EPISODE: #{best_idx}  "
            f"x_pos={best_ep.x_pos}  reward={best_ep.reward:.1f}  "
            f"time={best_start}-{best_end} ({duration:.1f}s)"
        )
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_summary_table(
    results: list[ModelResult], include_skipped: bool = True
) -> str:
    """Return a plain-text comparison table string."""
    header = (
        f"{'Model':<14} | {'Avg Reward':>11} | {'Avg Dist':>9} | "
        f"{'Max Dist':>9} | {'Complete%':>10} | {'Avg Steps':>10}"
    )
    sep = "-" * len(header)

    rows = [header, sep]

    for r in results:
        if r.skipped:
            if include_skipped:
                rows.append(
                    f"{r.key:<14} | {'SKIPPED':>11} | {'':>9} | {'':>9} | {'':>10} | {'':>10}"
                )
            continue
        rows.append(
            f"{r.key:<14} | {r.avg_reward:>11.1f} | {r.avg_distance:>9.1f} | "
            f"{r.max_distance:>9d} | {r.completion_rate:>9.1f}% | {r.avg_steps:>10.1f}"
        )

    return "\n".join(rows)


def format_markdown_table(results: list[ModelResult]) -> str:
    """Return a GitHub-flavoured markdown table."""
    header = (
        "| Model | Avg Reward | Avg Distance | Max Distance | "
        "Completion % | Avg Steps | Avg Score | Avg Coins |"
    )
    align = (
        "|:------|----------:|-------------:|-------------:|"
        "------------:|----------:|----------:|----------:|"
    )
    rows = [header, align]

    for r in results:
        if r.skipped:
            rows.append(f"| {r.key} | *skipped* | | | | | | |")
            continue
        rows.append(
            f"| {r.key} "
            f"| {r.avg_reward:.1f} "
            f"| {r.avg_distance:.1f} "
            f"| {r.max_distance} "
            f"| {r.completion_rate:.1f}% "
            f"| {r.avg_steps:.1f} "
            f"| {r.avg_score:.1f} "
            f"| {r.avg_coins:.1f} |"
        )

    return "\n".join(rows)


def format_per_model_detail(result: ModelResult) -> str:
    """Return a markdown section with per-episode breakdown for one model."""
    if result.skipped:
        return f"### {result.key}\n\n*Skipped: {result.skip_reason}*\n"

    lines = [f"### {result.key}\n"]
    lines.append(
        f"- Avg reward: {result.avg_reward:.1f}  \n"
        f"- Avg distance: {result.avg_distance:.1f}  \n"
        f"- Max distance: {result.max_distance}  \n"
        f"- Completion rate: {result.completion_rate:.1f}%  \n"
        f"- Avg steps: {result.avg_steps:.1f}  \n"
        f"- Avg score: {result.avg_score:.1f}  \n"
        f"- Avg coins: {result.avg_coins:.1f}  \n"
    )

    # Episode table
    lines.append("")
    lines.append("| Ep | Reward | X Pos | Score | Coins | Steps | Done |")
    lines.append("|---:|-------:|------:|------:|------:|------:|:-----|")
    for i, ep in enumerate(result.episodes, 1):
        done = "FLAG" if ep.flag_get else ("YES" if ep.x_pos >= 3150 else "-")
        lines.append(
            f"| {i} | {ep.reward:.1f} | {ep.x_pos} "
            f"| {ep.score} | {ep.coins} | {ep.steps} | {done} |"
        )

    return "\n".join(lines)


def write_markdown_report(
    results: list[ModelResult],
    output_path: Path,
    num_episodes: int,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        "# MLP Model Evaluation Report",
        "",
        f"**Date**: {timestamp}  ",
        f"**Episodes per model**: {num_episodes}  ",
        "**Game**: SuperMarioBros-v3 (World 1-1)  ",
        "**Completion threshold**: x_pos >= 3150 or flag_get  ",
        "",
        "## Summary",
        "",
        format_markdown_table(results),
        "",
        "---",
        "",
        "## Per-Model Detail",
        "",
    ]

    for r in results:
        sections.append(format_per_model_detail(r))
        sections.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nMarkdown report written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all trained MLP models on World 1-1."
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        metavar="N",
        help="Number of episodes per model (default: 50)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        metavar="KEY1,KEY2,...",
        help=(
            "Comma-separated list of model keys to evaluate. "
            f"Valid keys: {', '.join(REGISTRY_BY_KEY)}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/evaluation_results.md",
        metavar="PATH",
        help="Path to write the markdown report (default: docs/evaluation_results.md)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render in human mode (opens game window for screen recording)",
    )
    return parser.parse_args()


def resolve_model_specs(keys_arg: str | None) -> list[ModelSpec]:
    """Return the subset (or full list) of ModelSpecs to evaluate."""
    if keys_arg is None:
        return list(MODEL_REGISTRY)

    requested = [k.strip() for k in keys_arg.split(",") if k.strip()]
    unknown = [k for k in requested if k not in REGISTRY_BY_KEY]
    if unknown:
        print(f"[ERROR] Unknown model key(s): {', '.join(unknown)}")
        print(f"Valid keys: {', '.join(REGISTRY_BY_KEY)}")
        sys.exit(1)

    return [REGISTRY_BY_KEY[k] for k in requested]


def main() -> None:
    args = parse_args()
    specs = resolve_model_specs(args.models)
    output_path = PROJECT_ROOT / args.output

    print("\nMLP Model Evaluation")
    print(f"{'=' * 60}")
    print(f"Models to evaluate: {[s.key for s in specs]}")
    print(f"Episodes per model: {args.num_episodes}")
    print(f"Render mode:        {'human' if args.render else 'headless'}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")

    eval_start = time.time()
    print(f"Evaluation started at: {datetime.now().strftime('%H:%M:%S')}\n")

    results: list[ModelResult] = []
    for spec in specs:
        result = evaluate_model(spec, args.num_episodes, render=args.render)
        results.append(result)

    # Print summary table to stdout
    print(f"\n\n{'=' * 60}")
    print("  EVALUATION COMPLETE — COMPARISON TABLE")
    print(f"{'=' * 60}\n")
    print(format_summary_table(results))

    # Print best episode timestamps for video trimming
    print(f"\n{'=' * 60}")
    print("  BEST EPISODES — TIMESTAMPS FOR VIDEO TRIMMING")
    print(f"{'=' * 60}")
    eval_start_dt = datetime.fromtimestamp(eval_start)
    print(f"  Recording started: {eval_start_dt.strftime('%H:%M:%S')}\n")
    for r in results:
        if r.skipped or not r.episodes:
            continue
        best = max(r.episodes, key=lambda e: e.x_pos)
        best_idx = r.episodes.index(best) + 1
        offset_s = best.wall_start - eval_start
        offset_end = best.wall_end - eval_start
        mm_s, ss_s = divmod(int(offset_s), 60)
        hh_s, mm_s = divmod(mm_s, 60)
        mm_e, ss_e = divmod(int(offset_end), 60)
        hh_e, mm_e = divmod(mm_e, 60)
        print(
            f"  {r.key:<14}  ep#{best_idx:<3}  "
            f"x={best.x_pos:<5}  reward={best.reward:>8.1f}  "
            f"offset {hh_s:02d}:{mm_s:02d}:{ss_s:02d}-{hh_e:02d}:{mm_e:02d}:{ss_e:02d}"
        )
    print()

    # Write markdown report
    write_markdown_report(results, output_path, args.num_episodes)


if __name__ == "__main__":
    main()
