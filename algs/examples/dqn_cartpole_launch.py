import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from algs.dqn.dqn_agent import PytorchDqnAgent
from algs.launcher import learn_problem, result_learning


def main():
    env = gym.make('CartPole-v1')
    env2 = gym.make('CartPole-v1', render_mode="human")
    state_space = 4
    action_space = 2
    layers = [20, 12]
    agent = PytorchDqnAgent(state_space, action_space, layers)
    need_render = False
    episodes = 100
    max_steps = 200
    scores = learn_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env2, agent, max_steps)

if __name__ == '__main__':
    main()

