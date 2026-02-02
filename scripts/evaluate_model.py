"""
Evaluation script for trained Mario RL agent.
Loads a saved model and runs test episodes to measure performance.
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT

# Map string to action space for CLI argument
ACTION_SPACES = {
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
    "RIGHT_ONLY": RIGHT_ONLY,
}
from stable_baselines3 import DQN, PPO  # noqa: E402

from src.environments.mario_env import make_mario_env  # noqa: E402


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Trained Mario Agent")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Saved Model (e.g., models/dqn_baseline_world1-1_final.zip)",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of Evaluation Episodes to run (default: 10)",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        help="Which RL Alg Would you Like to Use?",
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=4,
        help="Frame skip value (must match training config)",
    )

    parser.add_argument(
        "--action-space",
        type=str,
        default="RIGHT_ONLY",
        choices=["SIMPLE_MOVEMENT", "RIGHT_ONLY"],
        help="Action space (must match training config, default: RIGHT_ONLY)",
    )

    parser.add_argument(
        "--game-version",
        type=str,
        default="SuperMarioBros-v0",
        help="Game version (v0 for better graphics, v3 for training, default: v0)",
    )

    parser.add_argument(
        "--reward-wrapper",
        type=str,
        default="standard",
        choices=["standard", "speedrun"],
        help="Reward wrapper (standard or speedrun, default: standard)",
    )

    parser.add_argument(
        "--observation-mode",
        type=str,
        default="pixel",
        choices=["pixel", "ram"],
        help="Observation mode (pixel for CnnPolicy, ram for MlpPolicy, default: pixel)",
    )

    return parser.parse_args()


def load_model(model_path, algorithm):
    print(f"🤖 Loading model from: {model_path}")
    if algorithm == "PPO":
        model = PPO.load(model_path)
    elif algorithm == "DQN":
        model = DQN.load(model_path)
    else:
        model = PPO.load(model_path)
    print("✅ Model loaded successfully!")
    return model


def create_eval_environment(
    skip=4,
    action_space="RIGHT_ONLY",
    game_version="SuperMarioBros-v0",
    reward_wrapper="standard",
    observation_mode="pixel",
):
    """Create Mario environment for evaluation with rendering enabled."""
    print("🎮 Creating Evaluation Environment")
    print(f"   Game: {game_version}")
    print(f"   Action Space: {action_space}")
    print(f"   Frame Skip: {skip}")
    print(f"   Reward Wrapper: {reward_wrapper}")
    print(f"   Observation Mode: {observation_mode}")

    # Convert string to action space
    action_space_obj = ACTION_SPACES[action_space]

    env = make_mario_env(
        game_version=game_version,
        action_space=action_space_obj,
        render_mode="human",
        skip=skip,
        reward_wrapper=reward_wrapper,
        observation_mode=observation_mode,
    )

    print("✅ Environment Created! Game Window Will Appear")
    return env


def evaluate_episode(model, env, episode_num):
    """
    Run one evaluation episode.

    Args:
        model: Trained DQN model
        env: Mario environment
        episode_num: Episode number (for printing)

    Returns:
        dict: Episode metrics (reward, distance, flag_get, etc.)
    """
    print(f"\n{'=' * 50}")
    print(f"Episode {episode_num}")
    print(f"{'=' * 50}")

    # Reset environment
    obs, info = env.reset()

    # Initialize tracking variables
    total_reward = 0
    steps = 0
    done = False

    # Episode loop - agent plays until done
    # Timeout at 5000 steps (reasonable for trained agent)
    while not done and steps < 5000:
        action, _ = model.predict(obs, deterministic=False)
        action = int(action)  # Convert numpy array to Python int
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    # Collect metrics from final state
    metrics = {
        "episode": episode_num,
        "reward": total_reward,
        "distance": info["x_pos"],
        "score": info["score"],
        "coins": info["coins"],
        "flag_get": info["flag_get"],
        "life": info["life"],
        "steps": steps,
    }

    # Print episode results
    print(f"\n📊 Episode {episode_num} Results:")
    print(f"   Reward: {metrics['reward']:.0f}")
    print(f"   Distance: {metrics['distance']} pixels")
    print(f"   Score: {metrics['score']}")
    print(f"   Steps: {metrics['steps']}")
    print(f"   Flag: {'✅ REACHED!' if metrics['flag_get'] else '❌ Not reached'}")

    return metrics


def main():
    """Main evaluation function."""
    print("\n" + "=" * 60)
    print("🎮 MARIO RL AGENT EVALUATION")
    print("=" * 60)

    # Parse arguments
    args = parse_args()

    # Load model
    model = load_model(args.model_path, args.algorithm)

    # Create environment
    env = create_eval_environment(
        skip=args.skip,
        action_space=args.action_space,
        game_version=args.game_version,
        reward_wrapper=args.reward_wrapper,
        observation_mode=args.observation_mode,
    )

    # Run evaluation episodes
    print(f"\n🏃 Running {args.num_episodes} evaluation episodes...")
    all_metrics = []

    for episode_num in range(1, args.num_episodes + 1):
        metrics = evaluate_episode(model, env, episode_num)
        all_metrics.append(metrics)

    # Calculate statistics
    print("\n📈 Calculating statistics...")

    # Rewards
    rewards = [m["reward"] for m in all_metrics]
    avg_reward = statistics.mean(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)

    # Distance
    distances = [m["distance"] for m in all_metrics]
    avg_distance = statistics.mean(distances)
    max_distance = max(distances)
    min_distance = min(distances)

    # Steps
    steps_list = [m["steps"] for m in all_metrics]
    avg_steps = statistics.mean(steps_list)
    max_steps = max(steps_list)
    min_steps = min(steps_list)

    # Score
    scores = [m["score"] for m in all_metrics]
    avg_score = statistics.mean(scores)
    max_score = max(scores)

    # Coins
    coins_list = [m["coins"] for m in all_metrics]
    total_coins = sum(coins_list)
    avg_coins = statistics.mean(coins_list)

    # Success count
    success_count = sum([m["flag_get"] for m in all_metrics])
    success_rate = (success_count / args.num_episodes) * 100

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print("\n🎯 Overall Performance:")
    print(f"   Episodes Run: {args.num_episodes}")
    print(f"   Success Rate: {success_count}/{args.num_episodes} ({success_rate:.1f}%)")

    print("\n🏆 Reward Stats:")
    print(f"   Average: {avg_reward:.1f}")
    print(f"   Max: {max_reward:.1f}")
    print(f"   Min: {min_reward:.1f}")

    print("\n📏 Distance Stats:")
    print(f"   Average: {avg_distance:.1f} pixels")
    print(f"   Max: {max_distance} pixels")
    print(f"   Min: {min_distance} pixels")
    print("   (Flag is at ~3266 pixels)")

    print("\n⏱️  Steps Stats:")
    print(f"   Average: {avg_steps:.1f}")
    print(f"   Max: {max_steps}")
    print(f"   Min: {min_steps}")

    print("\n🎮 Game Stats:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Max Score: {max_score}")
    print(f"   Total Coins: {total_coins}")
    print(f"   Avg Coins/Episode: {avg_coins:.1f}")

    print("\n" + "=" * 60)

    # Comparison to baseline (from Phase 2)
    print("\n📊 Comparison to Random Baseline:")
    print("   Random avg reward: ~380")
    print(f"   Trained avg reward: {avg_reward:.1f}")
    print(f"   Improvement: {(avg_reward / 380):.1f}x better!")
    print("\n   Random max distance: 434 pixels")
    print(f"   Trained max distance: {max_distance} pixels")
    print(f"   Improvement: {(max_distance / 434):.1f}x further!")

    # Comparison to DQN (from Phase 3)
    print("\n📊 Comparison to DQN Baseline (2M steps):")
    print("   DQN avg reward: ~1920")
    print(f"   PPO avg reward: {avg_reward:.1f}")
    if avg_reward > 1920:
        print(f"   🎉 PPO is {(avg_reward / 1920):.2f}x better!")
    else:
        print(f"   DQN was {(1920 / avg_reward):.2f}x better")
    print("\n   DQN avg distance: ~1024 pixels")
    print(f"   PPO avg distance: {avg_distance:.1f} pixels")
    print("\n   DQN max distance: ~1673 pixels")
    print(f"   PPO max distance: {max_distance} pixels")
    print("\n   DQN success rate: 0%")
    print(f"   PPO success rate: {success_rate:.1f}%")

    # Close environment
    env.close()

    print("\n✅ Evaluation complete!")
    print("💡 Tip: Use --num-episodes 20 for more reliable statistics")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
