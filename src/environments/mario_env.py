import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from src.environments.wrappers import (
    CompatibilityWrapper,
    SkipFrameWrapper,
    RewardShapingWrapper,
    GrayscaleWrapper,
    ResizeWrapper,
    FrameStackWrapper,
    TransposeWrapper,
)


def make_mario_env(
    game_version="SuperMarioBros-v3",
    action_space=SIMPLE_MOVEMENT,
    render_mode=None,
    skip=4,
):
    env = gym_super_mario_bros.make(
        game_version, apply_api_compatibility=True, render_mode=render_mode
    )
    env = JoypadSpace(env, action_space)
    env = CompatibilityWrapper(env)  # Handle old/new Gym API differences
    env = SkipFrameWrapper(env, skip=skip)
    env = RewardShapingWrapper(env)  # Encourage forward progress, penalize idle/death
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env)
    # NormalizeWrapper removed - let SB3 handle normalization internally
    env = FrameStackWrapper(env)
    env = TransposeWrapper(env)  # Convert (H,W,C) -> (C,H,W) for PyTorch

    return env
