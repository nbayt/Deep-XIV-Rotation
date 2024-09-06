import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchinfo import summary

import random

from collections import deque
import models.models as models

class DQN:
    def __init__(self, env, epsilon = 1.0, max_history = 20):
        self.env = env
        self.features = env.get_state_shape()[0]
        self.actions = env.get_max_actions()
        self.epsilon = epsilon
        self.history = deque([], max_history)

        self.model, self.optim, _ = models.construct_densenet(self.features, self.actions)
        self.model = self.model.to(torch.device('cpu'))
        print(summary(self.model, input_size=(8,self.features)))

    def add_to_history(self, event):
        self.history.append(event)

    def sample_from_history(self, batch_size):
        return random.sample(self.history, batch_size)
    
    def predict(self, states):
        states = states.to(torch.float32)
        states = states.to(torch.device('cpu'))
        self.model = self.model.to(torch.device('cpu'))
        
        with torch.no_grad():
            self.model.train(False)
            outputs = self.model(states)
            return outputs

    
    def get_action(self, states, e=0.0):
        outputs = self.predict(states)
        if np.random.uniform(0, 1) < e:
            return torch.from_numpy(np.random.randint(self.env.get_max_actions(), size=len(states)))
        else:
            return np.argmax(outputs, axis=1).to(torch.int32)
    
    def history_size(self):
        return len(self.history)

    def train(self):
        pass

    def learn(self):
        pass

    def test(self):
        pass