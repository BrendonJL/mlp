# Imports
import gym_super_mario_bros
import wandb
import imageio
from tqdm import tqdm

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

# Cleanup
env.close()
wandb.finish()

print("\n✅ Baseline complete! Videos saved to data/videos/baseline/")
