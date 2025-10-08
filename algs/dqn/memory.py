from collections import deque
import numpy as np
import random as rnd


class Memory:
    def __init__(self, max_len):
        self.memory = deque(maxlen=max_len)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = rnd.sample(self.memory, batch_size)
        batch = list(zip(*batch))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.memory)
