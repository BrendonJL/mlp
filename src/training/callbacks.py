import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from src.utils import db_logger


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.max_x_pos = []
        self.starting_stage = []
        self.accumulated_distance = []
        self.prev_stage = []
        self.prev_x_pos = []

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.max_x_pos = np.zeros(n_envs, dtype=np.float32)
        self.starting_stage = np.zeros(n_envs, dtype=np.int32)
        self.accumulated_distance = np.zeros(n_envs, dtype=np.float32)
        self.prev_stage = np.zeros(n_envs, dtype=np.int32)
        self.prev_x_pos = np.zeros(n_envs, dtype=np.float32)

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i in range(len(dones)):
            info = infos[i]
            current_x = info["x_pos"]
            current_stage = info["stage"]
            if current_stage > self.prev_stage[i]:
                self.accumulated_distance[i] += self.prev_x_pos[i]

            current_total = self.accumulated_distance[i] + current_x
            self.max_x_pos[i] = max(self.max_x_pos[i], current_total)

            self.prev_stage[i] = current_stage
            self.prev_x_pos[i] = current_x

            if dones[i]:
                episode_info = info.get("episode")
                if episode_info is not None:
                    final_x_pos = self.accumulated_distance[i] + current_x
                    level_completed = current_stage != self.starting_stage[i]
                    wandb.log(
                        {
                            "episode_reward": episode_info["r"],
                            "episode_length": episode_info["l"],
                            "episode_max_x_pos": self.max_x_pos[i],
                            "episode_final_x_pos": final_x_pos,
                            "episode_level_complete": level_completed,
                        }
                    )
                    self.accumulated_distance[i] = 0
                    self.max_x_pos[i] = 0
                    self.starting_stage[i] = current_stage
                    self.prev_stage[i] = current_stage
                    self.prev_x_pos[i] = current_x
        return True


class DatabaseCallback(BaseCallback):
    def __init__(self, experiment_id, verbose=0):
        super().__init__(verbose)
        self.experiment_id = experiment_id
        self.episode_count = 0
        self.accumulated_distance = []
        self.max_x_pos = []
        self.starting_stage = []
        self.prev_stage = []
        self.prev_x_pos = []

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.max_x_pos = np.zeros(n_envs, dtype=np.float32)
        self.starting_stage = np.zeros(n_envs, dtype=np.int32)
        self.accumulated_distance = np.zeros(n_envs, dtype=np.float32)
        self.prev_stage = np.zeros(n_envs, dtype=np.int32)
        self.prev_x_pos = np.zeros(n_envs, dtype=np.float32)

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i in range(len(dones)):
            info = infos[i]
            current_x = info["x_pos"]
            current_stage = info["stage"]
            if current_stage > self.prev_stage[i]:
                self.accumulated_distance[i] += self.prev_x_pos[i]

            current_total = self.accumulated_distance[i] + current_x
            self.max_x_pos[i] = max(self.max_x_pos[i], current_total)

            self.prev_stage[i] = current_stage
            self.prev_x_pos[i] = current_x

            if dones[i]:
                episode_info = info.get("episode")
                if episode_info is not None:
                    self.episode_count += 1
                    final_x_pos = self.accumulated_distance[i] + current_x
                    level_completed = current_stage != self.starting_stage[i]

                    episode_data = {
                        "episode_number": self.episode_count,
                        "total_reward": float(episode_info["r"]),
                        "episode_length": int(episode_info["l"]),
                        "x_pos": int(info["x_pos"]),
                        "max_x_pos": int(self.max_x_pos[i]),
                        "final_x_pos": int(final_x_pos),
                        "level_completed": bool(level_completed),
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

                    self.accumulated_distance[i] = 0
                    self.max_x_pos[i] = 0
                    self.starting_stage[i] = current_stage
                    self.prev_stage[i] = current_stage
                    self.prev_x_pos[i] = current_x

                    # Print reward breakdown if available (for debugging)
                    breakdown = info.get("reward_breakdown")
                    if breakdown:
                        print(
                            f"   💰 base={breakdown['base_game']}, fwd={breakdown['forward']}, "
                            f"back={breakdown['backward']}, idle={breakdown['idle']}, "
                            f"death={breakdown['death']}, mile={breakdown['milestone']}"
                        )
        return True
