import os

import numpy as np
import torch as T
from algs.ppo.networks import ActorNetwork, CriticNetwork
from algs.ppo.ppo_memory import PPOMemory


class Agent:
    def __init__(self, input_dims, n_actions, layers,
                 gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.1,
                 batch_size=64, n_epochs=4):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(n_actions, input_dims, alpha, layers)
        self.critic = CriticNetwork(input_dims, alpha, layers)
        self.memory = PPOMemory(batch_size)

    def store(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def store_ep(self, rewards):
        self.memory.store_ep(rewards)

    def store_val(self, values):
        self.memory.store_val(values)

    def store_dones(self, dones):
        self.memory.store_dones(dones)

    def choose_action(self, observation, deterministic=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)

        if deterministic:
            # Take the most likely action (mean for continuous, argmax for discrete)
            action = dist.probs.argmax().item()  # For discrete actions (like in CartPole)
        else:
            # Sample from the distribution (used during training)
            action = dist.sample().item()

        # Optional: log probability of the chosen action (useful for debugging)
        probs = T.squeeze(dist.log_prob(T.tensor([action]).to(self.actor.device))).item()
        value = T.squeeze(value).item()

        return action, probs, value


    def act(self, observation, deterministic=True):
        return self.choose_action(observation, deterministic)[0]


    def compute_discount_rewards(self, ep_rewards_arr):
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(ep_rewards_arr):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = np.array(batch_rtgs)

        return batch_rtgs

    def compute_gae_rewards(self, rewards, values, dones, lamb, gamma):
        batch_gae = []
        eps_len = len(rewards)

        for batch_id in range(eps_len):
            gae = 0.0
            r_batch = rewards[eps_len-batch_id-1]
            v_batch = values[eps_len-batch_id-1]
            d_batch = dones[eps_len-batch_id-1]
            batch_len = len(r_batch)
            for i in range(batch_len):
                r = r_batch[batch_len-i-1]
                v = v_batch[batch_len-i-1]
                v_next = 0.0
                if i > 0:
                    v_next = v_batch[batch_len-i]
                d = d_batch[batch_len-i-1]
                mask = int(1-d)

                delta = r + gamma * v_next * mask - v
                gae = delta + gamma * lamb * mask * gae
                batch_gae.insert(0, gae+v)

        return np.array(batch_gae)

    def learn(self, values_hist=None, adv_hist=None):
        state_arr, action_arr, old_probs_arr, vals_arr, \
        reward_arr, done_arr, ep_rewards_arr, ep_values_err, ep_dones_arr, batches = self.memory.generate_batches()
        disc_r_arr = self.compute_discount_rewards(ep_rewards_arr)
        # gae_r_arr = self.compute_gae_rewards(ep_rewards_arr, ep_values_err, ep_dones_arr, self.gae_lambda, self.gamma)

        advantage = disc_r_arr - vals_arr
        # advantage = gae_r_arr - vals_arr
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        if values_hist is not None:
            values_hist.append(vals_arr.mean())
        if adv_hist is not None:
            adv_hist.append(advantage.mean())

        advantage = T.tensor(advantage).to(self.actor.device)
        # values = T.tensor(vals_arr).to(self.actor.device)
        returns_arr = T.tensor(disc_r_arr).to(self.actor.device)
        for _ in range(self.n_epochs):
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                # prob_ratio = new_probs.exp() / old_probs.exp()
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[
                    batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = returns_arr[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                entropy = -dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()
