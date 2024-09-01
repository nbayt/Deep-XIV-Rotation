import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DQN:
    def __init__(self, model, env, epsilon = 1.0):
        self.model = model
        self.env = env
        self.epsilon = epsilon

    def train(self):
        pass

    def learn(self):
        pass

    def test(self):
        pass