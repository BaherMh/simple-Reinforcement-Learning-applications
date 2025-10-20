def learn_problem(env, agent, episodes, max_steps, need_render, callback=None):
    scores = []

    for e in range(episodes):
        state, _ = env.reset()
        score = 0

        for i in range(max_steps):
            if need_render:
                env.render()

            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            score += reward

            # Store experience
            agent.memory.remember(state, action, reward, next_state, int(done))
            state = next_state

            # Train from replay (mini-batch learning)
            agent.replay()

            if done:
                break

        # End of episode cleanup (e.g., decay epsilon, etc.)
        agent.end_episode()

        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)

        # 🔁 Call callback if provided
        if callback is not None:
            callback(e, score)

    return scores


def learn_policy_problem(env, agent, episodes, max_steps, N, need_render, callback=None):
    scores = []
    j = 0
    values = []  # Note: currently unused — might be intended for future use
    adv = []     # Same here — likely filled during learning

    for e in range(episodes):
        state, _ = env.reset()
        state = state  # / 255.0
        score = 0
        ep_rewards = []
        ep_values = []
        ep_dones = []

        for i in range(max_steps):
            if need_render:
                env.render()

            action, prob, val = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state  # / 255.0
            score += reward
            ep_rewards.append(reward)
            ep_values.append(val)
            ep_dones.append(done)
            j += 1

            agent.store(state, action, prob, val, reward, done)

            state = next_state
            if done:
                break

        # Store episode data for PPO learning
        agent.store_ep(ep_rewards)
        agent.store_val(ep_values)
        agent.store_dones(ep_dones)

        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)

        if callback is not None:
            callback(e, score, scores)  # You can pass more data if needed

        # Learn every N steps (not episodes)
        if j >= N:
            j = 0
            agent.learn(values, adv)  # Note: 'values' and 'adv' are empty — may need fix

    return scores, values, adv

import gymnasium as gym
import imageio
import numpy as np
from IPython.display import Video, display


def result_learning(env, agent, max_steps, num_episodes=5):
    """
    Run trained agent in environment, record videos of episodes, and display them.
    
    Args:
        env: The environment (should support render_mode='rgb_array')
        agent: Trained agent with .act() method
        max_steps: Maximum steps per episode
        num_episodes: Number of episodes to run and save
    """
    # Ensure the environment uses rgb_array for rendering

    agent.epsilon = 0.0  # No exploration
    all_frames = []
    scores = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        score = 0
        frames = []  # Store frames for this episode
        
        for step in range(max_steps):
            # Render to RGB array and store
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            state = next_state
            
            if done:
                break

        print(f"Episode {ep + 1}: Score = {score}")
        scores.append(score)
        all_frames.extend(frames)  # Add all frames from this episode

    env.close()

    # Save as MP4
    print("\nSaving evaluation video...")
    imageio.mimwrite('evaluation.mp4', np.array(all_frames), fps=30)

    # Display video in Colab
    display(Video('evaluation.mp4', embed=True))

    print(f"\nAverage score over {num_episodes} episodes: {np.mean(scores):.2f}")
