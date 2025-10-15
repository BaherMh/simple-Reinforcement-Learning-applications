from typing import Dict
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

class CustomCallbacks(DefaultCallbacks):



    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        episode.user_data["pole_angles"] = []
        episode.user_data["pole_angles_vel"] = []
        episode.user_data["episode_side_rate"] = 0

    def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        obs = episode.last_info_for()['obs']
        pole_angle = abs(obs[2])
        pole_angles_vel = abs(obs[3])
        episode.user_data["pole_angles"].append(pole_angle)
        episode.user_data["pole_angles_vel"].append(pole_angles_vel)
        episode.user_data["episode_side_rate"] += np.sign(obs[0]) == np.sign(obs[4] * 2 - 1)

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        pole_angle = np.mean(episode.user_data["pole_angles"])
        pole_angles_vel = np.mean(episode.user_data["pole_angles_vel"])
        episode.custom_metrics["pole_angles"] = pole_angle
        episode.custom_metrics["pole_angles_vel"] = pole_angles_vel
        episode.custom_metrics["episode_side_rate"] = episode.user_data["episode_side_rate"] / episode.length
        episode.hist_data["pole_angles"] = [pole_angle]
        episode.hist_data["pole_angles_vel"] = [pole_angles_vel]
        episode.hist_data["episode_side_rate"] = [episode.user_data["episode_side_rate"]]
        if episode.last_info_for()['obs'][4] == -1.0:
            episode.custom_metrics["reward_left"] = episode.total_reward
            episode.hist_data["reward_left"] = [episode.total_reward]
        else:
            episode.custom_metrics["reward_right"] = episode.total_reward
            episode.hist_data["reward_right"] = [episode.total_reward]