from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("LunarLander-v3")
)

stop = {
    "training_iteration": 5000,
    "timesteps_total": 500000,
    "episode_reward_mean": 200,
}

tune.run(
    "PPO",
    config=config.to_dict(),
    stop=stop,
    verbose=3,
    checkpoint_freq=100,
    checkpoint_at_end=True,
    # storage_path="./"
)
