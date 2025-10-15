import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import torch as T


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, layers):
        super().__init__()

        self.input = nn.Linear(*input_dims, layers[0])
        self.hidden = nn.Linear(layers[0], layers[1])
        self.output = nn.Linear(layers[1], n_actions)

        # w = np.sqrt(2)
        f = 0.1
        T.nn.init.orthogonal_(self.input.weight.data, f)
        T.nn.init.orthogonal_(self.hidden.weight.data, f)
        T.nn.init.orthogonal_(self.output.weight.data, f)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.input(state)
        x = T.relu(x)
        x = self.hidden(x)
        x = T.relu(x)
        x = self.output(x)
        dist = T.softmax(x, dim=-1)
        dist = Categorical(dist)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, layers):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value