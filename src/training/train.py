import argparse
import subprocess
import sys
import torch
import wandb
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from src.utils.config_loader import load_config
from src.utils import db_logger
from src.environments.mario_env import make_mario_env
from src.training.callbacks import WandbCallback, DatabaseCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Parse Arguments")

    parser.add_argument("--config", type=str, required=True, help="Cry about it")
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Still Cry About it?"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None, help="probably should still cry"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.experiment_name is not None:
        config["experiment"]["name"] = args.experiment_name

    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = args.total_timesteps

    # Collect metadata for reproducibility
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    pytorch_version = torch.__version__

    # Print experiment info
    print("=" * 60)
    print(f"🎮 Starting Training: {config['experiment']['name']}")
    print(f"📁 Config: {args.config}")
    print(f"⏱️  Total timesteps: {config['training']['total_timesteps']}")
    print(f"🔖 Git commit: {git_hash[:8]}")
    print(f"🐍 Python: {python_version}")
    print(f"🔥 PyTorch: {pytorch_version}")
    print("=" * 60)

    wandb.init(
        project=config["wandb"]["project"],
        name=config["experiment"]["name"],
        config=config,
        tags=config["wandb"]["tags"],
    )

    experiment_id = db_logger.create_experiment(
        experiment_name=config["experiment"]["name"],
        algorithm=config["experiment"]["algorithm"],
        git_commit_hash=git_hash,
        python_version=python_version,
        pytorch_version=pytorch_version,
        notes=config["experiment"].get("description", None),
    )
    print(f"📊 Created experiment ID: {experiment_id}")

    hyperparams = config["dqn_hyperparameters"]
    db_logger.log_hyperparameters(experiment_id, hyperparams)

    print("🎮 Creating Mario Environment")
    env = make_mario_env(
        game_version=config["environment"]["game"], action_space=SIMPLE_MOVEMENT
    )

    print("🤖 Starting DQN Agent")
    model = DQN(
        policy=config["dqn_hyperparameters"]["policy"],
        env=env,
        learning_rate=config["dqn_hyperparameters"]["learning_rate"],
        buffer_size=config["dqn_hyperparameters"]["buffer_size"],
        learning_starts=config["dqn_hyperparameters"]["learning_starts"],
        batch_size=config["dqn_hyperparameters"]["batch_size"],
        gamma=config["dqn_hyperparameters"]["gamma"],
        target_update_interval=config["dqn_hyperparameters"]["target_update_interval"],
        exploration_fraction=config["dqn_hyperparameters"]["exploration_fraction"],
        exploration_final_eps=config["dqn_hyperparameters"]["exploration_final_eps"],
        verbose=1,
    )
    print("✅ DQN Agent Created!")

    print("📡 Setting Up Callbacks")
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_freq"],
        save_path=config["paths"]["model_dir"],
        name_prefix=config["experiment"]["name"],
    )

    wandb_callback = WandbCallback(verbose=1)
    db_callback = DatabaseCallback(experiment_id=experiment_id, verbose=1)
    callbacks = [checkpoint_callback, wandb_callback, db_callback]

    print("🚀 Starting Training")
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=callbacks,
        log_interval=4,  # Print stats every 4 episodes
        progress_bar=True,  # Show tqdm progress bar for training
    )

    final_model_path = (
        f"{config['paths']['model_dir']}/{config['experiment']['name']}_final"
    )
    model.save(final_model_path)
    print(f"💾 Saved final model to {final_model_path}")

    db_logger.update_experiment(experiment_id, "completed", 0)

    env.close()
    wandb.finish()
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
