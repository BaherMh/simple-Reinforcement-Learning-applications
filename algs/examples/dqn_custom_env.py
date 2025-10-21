import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.ion()

from algs.dqn.dqn_agent import PytorchDqnAgent
from algs.launcher import learn_problem, result_learning
from examples.custom_env.cartpole_leftright_env import CartPoleLeftRightEnv
from IPython.display import clear_output


def main():
    env = CartPoleLeftRightEnv(config={"reward_fn": "smooth"})
    state_space = 5
    action_space = 2
    layers = [40, 24]
    agent = PytorchDqnAgent(state_space, action_space, layers)
    
    need_render = False
    episodes = 2000
    max_steps = 500

    # Setup live plot with dynamic scaling
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-', label="Episode Score")

    # 🔁 Remove fixed limits — let matplotlib auto-scale
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("DQN Training - Live Score")
    ax.legend()
    plt.tight_layout()

    scores = []

    def callback(ep_idx, score):
        scores.append(score)
        
        # Update plot data
        line.set_data(range(len(scores)), scores)
        ax.relim()
        ax.autoscale_view()

        # Force redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Optional: throttle updates to every N episodes for smoother performance
        if ep_idx % 1 == 0:  # Update every 10 episodes
            clear_output(wait=True)
            plt.pause(0.01)  # Small pause to allow GUI to catch up

    # Train with dynamic plot
    _ = learn_problem(env, agent, episodes, max_steps, need_render, callback=callback)

    plt.ioff()
    plt.show()

    result_learning(env, agent, max_steps)


if __name__ == '__main__':
    main()