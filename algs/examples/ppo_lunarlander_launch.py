import gymnasium as gym
from algs.ppo.ppo_agent import Agent
from algs.launcher import learn_problem, result_learning, learn_policy_problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def main():
    env = gym.make('LunarLander-v3', render_mode="human")
    env.metadata['render_fps'] = 1000
    N = 2048
    batch_size = 256
    n_epochs = 10
    alpha = 0.0003
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    layers = [64, 64]
    agent = Agent(n_actions=action_space, batch_size=batch_size, input_dims=state_space,
                  layers=layers, alpha=alpha, n_epochs=n_epochs)
    need_render = True
    episodes = 2000
    max_steps = 500
    scores, values, adv = learn_policy_problem(env, agent, episodes, max_steps, N, need_render)
    plt.figure()
    plt.plot(scores)
    # plt.figure()
    # plt.plot(values)
    # plt.figure()
    # plt.plot(adv)
    plt.show()
    result_learning(env, agent, max_steps)

if __name__ == '__main__':
    main()
