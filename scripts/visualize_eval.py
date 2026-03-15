"""
Visualization script for final evaluation results.

Parses evaluation_results.md and generates:
1. Violin plot: x_pos distribution per model
2. Bar chart: completion rate per model
3. Line plot: average distance progression
4. Bar chart: average steps for completing models (v5 vs v9)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "docs" / "evaluation_results.md"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "images" / "eval"

MODEL_ORDER = [
    "random",
    "dqn_baseline",
    "ppo_baseline",
    "ppo_v2",
    "ppo_v3",
    "ppo_v4",
    "ppo_v5",
    "ppo_v7",
    "ppo_v8",
    "ppo_v9",
]


def parse_results(filepath: Path) -> pd.DataFrame:
    read_text = filepath.read_text(encoding="utf-8")
    sections = read_text.split("###")
    rows = []
    for section in sections[1:]:
        lines = section.split("\n")
        model_name = lines[0].strip()
        for line in lines:
            if line.startswith("|") and "Ep" not in line and "---" not in line:
                cols = [c.strip() for c in line.split("|")[1:-1]]
                row = {
                    "model": model_name,
                    "episode": int(cols[0]),
                    "reward": float(cols[1]),
                    "x_pos": int(cols[2]),
                    "score": int(cols[3]),
                    "coins": int(cols[4]),
                    "steps": int(cols[5]),
                    "completed": cols[6] != "-",
                }
                rows.append(row)
    return pd.DataFrame(rows)


def plot_violin_xpos(df: pd.DataFrame) -> None:
    """Violin plot showing x_pos distribution per model."""
    _fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=df,
        x="model",
        y="x_pos",
        order=MODEL_ORDER,
        inner="box",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Distance Distribution by Model (x_pos)", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("X Position (distance)")
    ax.axhline(
        y=3150, color="red", linestyle="--", alpha=0.5, label="Completion (3150)"
    )
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "violin_xpos.png", dpi=150)
    plt.close()
    print("  Saved violin_xpos.png")


def plot_completion_rate(df: pd.DataFrame) -> None:
    """Bar chart of completion rate per model."""
    completion = (
        df.groupby("model")["completed"].mean().reindex(MODEL_ORDER).fillna(0) * 100
    )

    _fig, ax = plt.subplots(figsize=(12, 5))
    colors = [
        "#2ecc71" if v == 100 else "#e74c3c" if v == 0 else "#f39c12"
        for v in completion
    ]
    completion.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Level Completion Rate by Model", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_ylim(0, 110)

    for i, v in enumerate(completion):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "completion_rate.png", dpi=150)
    plt.close()
    print("  Saved completion_rate.png")


def plot_distance_progression(df: pd.DataFrame) -> None:
    """Line plot showing average distance progression across model versions."""
    avg_dist = df.groupby("model")["x_pos"].mean().reindex(MODEL_ORDER)

    _fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        range(len(MODEL_ORDER)),
        avg_dist.values,
        "o-",
        color="#3498db",
        linewidth=2,
        markersize=8,
    )
    ax.fill_between(
        range(len(MODEL_ORDER)), avg_dist.values, alpha=0.15, color="#3498db"
    )
    ax.axhline(
        y=3150, color="red", linestyle="--", alpha=0.5, label="Completion (3150)"
    )
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
    ax.set_title("Average Distance Progression Across Models", fontsize=14)
    ax.set_xlabel("Model Version")
    ax.set_ylabel("Average X Position")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_progression.png", dpi=150)
    plt.close()
    print("  Saved distance_progression.png")


def plot_efficiency_comparison(df: pd.DataFrame) -> None:
    """Bar chart comparing average steps for models that complete the level."""
    completing_models = ["ppo_v5", "ppo_v9"]
    completed_runs = df[(df["model"].isin(completing_models)) & (df["completed"])]
    avg_steps = (
        completed_runs.groupby("model")["steps"].mean().reindex(completing_models)
    )

    _fig, ax = plt.subplots(figsize=(6, 5))
    avg_steps.plot(
        kind="bar",
        ax=ax,
        color=["#9b59b6", "#2ecc71"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("Completion Efficiency: v5 vs v9", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Steps to Complete")

    for i, v in enumerate(avg_steps):
        ax.text(i, v + 10, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "efficiency_comparison.png", dpi=150)
    plt.close()
    print("  Saved efficiency_comparison.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing evaluation results...")
    df = parse_results(RESULTS_FILE)
    print(f"  Loaded {len(df)} episodes across {df['model'].nunique()} models\n")

    print("Generating visualizations...")
    plot_violin_xpos(df)
    plot_completion_rate(df)
    plot_distance_progression(df)
    plot_efficiency_comparison(df)

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
