import os
import pickle
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
import gymnasium as gym

def create_env(env_config):
    env = gym.make("LunarLander-v2", render_mode="human")
    env.metadata['render_fps'] = 500
    return env

def read_config(checkpoint_path) -> dict:
    with open(os.path.join(checkpoint_path, "algorithm_state.pkl"), 'rb') as file:
        config = pickle.load(file)
    return config

checkpoint_path = ""

ray.init(local_mode=True)

exp_config = read_config(checkpoint_path)['config']
exp_config['num_workers'] = 0

tune.register_env(
    "my_env",
    create_env,
)
agent = PPO(exp_config)
agent.load_checkpoint(checkpoint_path)
policy = agent.get_policy()

env = create_env({})
for _ in range(1000):
    score = 0.0
    state = env.reset()[0]
    for _ in range(300):
        act = agent.compute_single_action(state)
        state, r, d, tr, i = env.step(act)
        score += r
        if d:
            break
    print("Score: " + str(score))
ray.shutdown()
