"""Random agent baseline for Mario RL."""

import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

import gym_super_mario_bros  # noqa: E402
import imageio  # noqa: E402
import wandb  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.utils import db_logger  # noqa: E402

wandb.init(
    project="mario-rl-agent",
    name="random-baseline",
    config={"num_episodes": 10, "max_steps_per_episode": 1000, "algorithm": "random"},
    tags=["baseline", "random-agent", "phase-2"],
)

# Create the Environment (with render mode for video capture)
env = gym_super_mario_bros.make(
    "SuperMarioBros-v3", apply_api_compatibility=True, render_mode="rgb_array"
)

# Reset environment to get initial observation
observation = env.reset()

# Get version metadata
try:
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    git_hash = "unknown"

python_version = platform.python_version()

try:
    import torch

    pytorch_version = torch.__version__
except ImportError:
    pytorch_version = "N/A"

# Create database experiment
experiment_id = db_logger.create_experiment(
    experiment_name="random_baseline_world1-1",
    algorithm="random",
    git_commit_hash=git_hash,
    python_version=python_version,
    pytorch_version=pytorch_version,
    notes="Random baseline for comparison with DQN agent",
)

# Log hyperparameters
db_logger.log_hyperparameters(
    experiment_id=experiment_id,
    hyperparams_dict={
        "num_episodes": 10,
        "max_steps_per_episode": 1000,
        "algorithm": "random",
    },
)

# Episode Config
num_episode = 10
max_steps_per_episode = 1000
episodes_to_record = [0, 3, 6]  # Record episodes 0, 3, and 6

# Episode Loop with progress bar
for episode in tqdm(range(num_episode), desc="Random Baseline Episodes"):
    observation = env.reset()

    done = False
    total_reward = 0
    step = 0

    # Collect frames if this episode should be recorded
    frames = [] if episode in episodes_to_record else None

    # Game Loop
    while not done and step < max_steps_per_episode:
        # Capture frame for video if recording this episode
        if frames is not None:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception:
                pass  # Skip frame if render fails

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

    # Save video if we recorded this episode
    if frames is not None and len(frames) > 0:
        video_path = f"data/videos/baseline/random_baseline_episode_{episode}.mp4"
        try:
            imageio.mimsave(video_path, frames, fps=30)
            print(f"  → Saved video: {video_path} ({len(frames)} frames)")
        except Exception as e:
            print(f"  → Failed to save video for episode {episode}: {e}")

    # Extract final episode metrics
    final_x_pos = info["x_pos"]
    final_y_pos = info["y_pos"]
    final_score = info["score"]
    final_time = info["time"]
    final_coins = info["coins"]
    final_life = info["life"]
    final_status = info["status"]
    final_flag_get = info["flag_get"]
    final_world = info["world"]
    final_stage = info["stage"]

    # Log all metrics to wandb
    wandb.log(
        {
            "episode": episode + 1,
            "reward": total_reward,
            "episode_length": step,
            "x_pos": final_x_pos,
            "y_pos": final_y_pos,
            "time": final_time,
            "coins": final_coins,
            "life": final_life,
            "status": final_status,
            "world": final_world,
            "stage": final_stage,
            "score": final_score,
            "flag_get": final_flag_get,
        }
    )

    # Log episode to database
    episode_data = {
        "episode_number": episode + 1,
        "total_reward": float(total_reward),
        "episode_length": int(step),
        "x_pos": int(final_x_pos),
        "y_pos": int(final_y_pos),
        "time": int(final_time),
        "coins": int(final_coins),
        "life": int(final_life),
        "status": str(final_status),
        "world": int(final_world),
        "stage": int(final_stage),
        "score": int(final_score),
        "flag_get": bool(final_flag_get),
    }
    db_logger.log_episode(experiment_id=experiment_id, episode_data=episode_data)

# Update experiment with final status
db_logger.update_experiment(
    experiment_id=experiment_id, status="completed", total_episodes=num_episode
)

# Cleanup
env.close()
wandb.finish()
db_logger.close_connection_pool()

print("\n✅ Baseline complete! Videos saved to data/videos/baseline/")
