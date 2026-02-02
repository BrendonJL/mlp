from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.environments.mario_env import make_mario_env


def make_vec_mario_env(
    n_envs,
    game_version,
    action_space,
    skip=4,
    reward_wrapper="standard",
    observation_mode="pixel",
):
    def make_env(_):
        def _init():
            env = make_mario_env(
                game_version,
                action_space,
                skip=skip,
                reward_wrapper=reward_wrapper,
                observation_mode=observation_mode,
            )
            return env

        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])  # type: ignore[arg-type]
    vec_env = VecMonitor(vec_env)  # Track episode statistics for all envs
    return vec_env
