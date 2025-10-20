import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from algs.launcher import learn_policy_problem, learn_problem, result_learning
from algs.ppo.ppo_agent import Agent

# Use a backend that supports dynamic plotting
matplotlib.use("TkAgg")
plt.ion()  # Turn on interactive mode

from examples.custom_env.cartpole_leftright_env import CartPoleLeftRightEnv


def main():
    env = CartPoleLeftRightEnv(config={"reward_fn": "smooth"})
    N = 2048
    batch_size = 256
    n_epochs = 10
    alpha = 0.0003
    state_space = (5,)
    action_space = 2
    layers = [64, 64]
    agent = Agent(n_actions=action_space, batch_size=batch_size, input_dims=state_space,
                  layers=layers, alpha=alpha, n_epochs=n_epochs)
    
    need_render = False
    episodes = 2000
    max_steps = 500

    # Setup for online plotting
    fig, ax = plt.subplots()
    scores = []
    line, = ax.plot(scores)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Training Scores')
    plt.tight_layout()

    def update_plot(score):
        scores.append(score)
        line.set_data(range(len(scores)), scores)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Modified version of learn_policy_problem to allow online updates
    # Or wrap the original function with a callback
    from functools import partial

    def callback(ep_idx, score, *args):
        print(f"Episode {ep_idx}: Score = {score}")
        update_plot(score)

    # Assuming your learn_policy_problem supports a callback or can be modified
    # If not, you'll need to modify learn_policy_problem to call the callback after each episode
    scores_list, values, adv = learn_policy_problem(
        env, agent, episodes, max_steps, N, need_render, callback=partial(callback)
    )

    # Keep the plot open after training
    plt.ioff()
    plt.show()

    result_learning(env, agent, max_steps)


if __name__ == '__main__':
    main()