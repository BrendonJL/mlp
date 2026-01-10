from stable_baselines3.common.vec_env import SubprocVecEnv
from src.environments.mario_env import make_mario_env


def make_vec_mario_env(n_envs, game_version, action_space):
    def make_env(_):
        def _init():
            env = make_mario_env(game_version, action_space)
            return env

        return _init

    return SubprocVecEnv([make_env(i) for i in range(n_envs)])
