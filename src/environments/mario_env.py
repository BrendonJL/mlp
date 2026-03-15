from typing import Any

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.environments.wrappers import (
    CompatibilityWrapper,
    FrameStackWrapper,
    GrayscaleWrapper,
    RAMObservationWrapper,
    ResizeWrapper,
    RewardShapingWrapper,
    SkipFrameWrapper,
    SpeedrunRewardWrapper,
    TransposeWrapper,
)

# Map config string to action space
ACTION_SPACES = {
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
    "RIGHT_ONLY": RIGHT_ONLY,
}


def make_mario_env(
    game_version: str = "SuperMarioBros-v3",
    action_space: list = SIMPLE_MOVEMENT,
    render_mode: str | None = None,
    skip: int = 4,
    reward_wrapper: str = "standard",
    observation_mode: str = "pixel",
) -> Any:
    env = gym_super_mario_bros.make(
        game_version, apply_api_compatibility=True, render_mode=render_mode
    )
    env = JoypadSpace(env, action_space)
    env = CompatibilityWrapper(env)  # Handle old/new Gym API differences
    env = SkipFrameWrapper(env, skip=skip)
    if reward_wrapper == "speedrun":
        env = SpeedrunRewardWrapper(env)
    else:
        env = RewardShapingWrapper(env)

    # Apply observation wrappers based on mode
    if observation_mode == "ram":
        # RAM-based grid observations for MlpPolicy
        env = RAMObservationWrapper(env)
    else:
        # Pixel-based observations for CnnPolicy
        env = GrayscaleWrapper(env)
        env = ResizeWrapper(env)
        # NormalizeWrapper removed - let SB3 handle normalization internally
        env = FrameStackWrapper(env)
        env = TransposeWrapper(env)  # Convert (H,W,C) -> (C,H,W) for PyTorch

    return env
