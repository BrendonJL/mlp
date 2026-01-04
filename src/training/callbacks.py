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

                # Construct episode_data dictionary for db_logger
                # Convert NumPy types to Python natives for PostgreSQL
                episode_data = {
                    "episode_number": self.episode_count,
                    "total_reward": float(episode_info["r"]),
                    "episode_length": int(episode_info["l"]),
                    "x_pos": int(info["x_pos"]),
                    "y_pos": int(info["y_pos"]),
                    "time": int(info["time"]),
                    "coins": int(info["coins"]),
                    "life": int(info["life"]),
                    "status": str(info["status"]),
                    "world": int(info["world"]),
                    "stage": int(info["stage"]),
                    "score": int(info["score"]),
                    "flag_get": bool(info["flag_get"]),
                }

                db_logger.log_episode(
                    experiment_id=self.experiment_id, episode_data=episode_data
                )

        return True
