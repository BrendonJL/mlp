"""
Recovery script for broken SB3 model files.

Extracts neural network weights from old model zips and rebuilds them
as fresh SB3-compatible model files using the current environment.
"""

import io
import zipfile
from pathlib import Path

import torch
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from stable_baselines3 import PPO

from src.environments.mario_env import make_mario_env

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_model_config(model_key: str) -> dict | None:
    configs = {
        "ppo_v3": {
            "policy": "CnnPolicy",
            "action_space": SIMPLE_MOVEMENT,
            "obs_mode": "pixel",
        },
        "ppo_v4": {
            "policy": "CnnPolicy",
            "action_space": SIMPLE_MOVEMENT,
            "obs_mode": "pixel",
        },
        "ppo_v5": {
            "policy": "CnnPolicy",
            "action_space": RIGHT_ONLY,
            "obs_mode": "pixel",
        },
        "ppo_v7": {
            "policy": "MlpPolicy",
            "action_space": SIMPLE_MOVEMENT,
            "obs_mode": "ram",
        },
    }
    return configs.get(model_key)


def recover_model(model_key: str, original_path: Path, output_path: Path) -> bool:
    """Extract weights from broken model zip and rebuild as fresh SB3 model."""
    config = get_model_config(model_key)
    if config is None:
        print(f"  Skipping {model_key}: no config defined")
        return False

    # Extract weights from old zip
    print(f"  Extracting weights from {original_path.name}...")
    with zipfile.ZipFile(original_path) as z, z.open("policy.pth") as f:
        old_weights = torch.load(
            io.BytesIO(f.read()), map_location="cpu", weights_only=False
        )

    # Create fresh environment with matching config
    env = make_mario_env(
        game_version="SuperMarioBros-v3",
        action_space=config["action_space"],
        render_mode=None,
        observation_mode=config["obs_mode"],
    )

    # Create fresh PPO with matching architecture
    fresh_model = PPO(
        policy=config["policy"],
        env=env,
        verbose=0,
    )

    # Load old weights into fresh model
    print(f"  Loading weights into fresh {config['policy']}...")
    fresh_model.policy.load_state_dict(old_weights)

    # Save as clean model file
    fresh_model.save(str(output_path))
    print(f"  Saved recovered model to {output_path.name}")

    env.close()
    return True


def main():
    models_dir = PROJECT_ROOT / "models"

    # Models that need recovery (segfault on PPO.load)
    broken_models = {
        "ppo_v3": "ppo_v3_world1-1_final.zip",
        "ppo_v4": "ppo_v4_world1-1_final.zip",
        "ppo_v5": "ppo_v5_world1-1_final.zip",
        "ppo_v7": "ppo_v7_world1-1_final.zip",
    }

    print("=" * 60)
    print("  Model Recovery Script")
    print("=" * 60)

    for key, filename in broken_models.items():
        original = models_dir / filename
        if not original.exists():
            print(f"\n[SKIP] {key}: {filename} not found")
            continue

        # Save recovered model with _recovered suffix
        recovered = models_dir / filename.replace("_final.zip", "_recovered.zip")
        print(f"\n[RECOVERING] {key}")

        try:
            success = recover_model(key, original, recovered)
            if success:
                print(f"  [OK] {key} recovered successfully")
        except Exception as e:
            print(f"  [FAIL] {key}: {e}")

    print("\n" + "=" * 60)
    print("  Recovery complete!")
    print("  Recovered models saved with '_recovered' suffix.")
    print(
        "  Test with: poetry run python -c \"from stable_baselines3 import PPO; PPO.load('models/<name>_recovered.zip')\""
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
