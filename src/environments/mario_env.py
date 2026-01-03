import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from src.environments.wrappers import (
    GrayscaleWrapper,
    ResizeWrapper,
    NormalizeWrapper,
    FrameStackWrapper,
)


def make_mario_env(game_version="SuperMarioBros-v3", action_space=SIMPLE_MOVEMENT):
    env = gym_super_mario_bros.make(game_version, apply_api_compatibility=True)
    env = JoypadSpace(env, action_space)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = FrameStackWrapper(env)

    return env
