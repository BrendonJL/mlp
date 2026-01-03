import wandb
from src.utils import db_logger
from stable_baselines3.common.callbacks import BaseCallback


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            episode_info = self.locals["infos"][0].get("episode")

            if episode_info is not None:
                wandb.log(
                    {
                        "episode_reward": episode_info["r"],
                        "episode_length": episode_info["l"],
                    }
                )

        return True


class DatabaseCallback(BaseCallback):
    def __init__(self, experiment_id, verbose=0):
        super().__init__(verbose)
        self.experiment_id = experiment_id
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            episode_info = self.locals["infos"][0].get("episode")

            if episode_info is not None:
                self.episode_count += 1
                info = self.locals["infos"][0]

                db_logger.log_episode(
                    experiment_id=self.experiment_id,
                    episode_number=self.episode_count,
                    reward=episode_info["r"],
                    episode_length=episode_info["l"],
                    x_pos=info["x_pos"],
                    y_pos=info["y_pos"],
                    time=info["time"],
                    coins=info["coins"],
                    life=info["life"],
                    status=info["status"],
                    world=info["world"],
                    stage=info["stage"],
                    score=info["score"],
                    flag_get=info["flag_get"],
                )

        return True
