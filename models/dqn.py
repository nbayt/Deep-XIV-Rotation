import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import random

from collections import deque

class DQN:
    def __init__(self, model, env, epsilon = 1.0, max_history = 20):
        self.model = model
        self.env = env
        self.epsilon = epsilon
        self.history = deque([], max_history)

    def add_to_history(self, event):
        self.history.append(event)

    def sample_from_history(self, batch_size):
        return random.sample(self.history, batch_size)
    
    def history_size(self):
        return len(self.history)

    def train(self):
        pass

    def learn(self):
        pass

    def test(self):
        pass