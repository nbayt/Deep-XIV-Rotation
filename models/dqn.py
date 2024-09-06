import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchinfo import summary

import random

from collections import deque
import models.models as models
import envs.base_env as baseenv

class DQN:
    def __init__(self, env: baseenv.BaseEnv, epsilon = 1.0, max_history = 20):
        self.env = env
        self.features = env.get_state_shape()
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

    def train(self, num_epochs=1, num_episodes_per_learning_session=10, session_limit=5):
        for epoch in range(num_epochs):
            num_sessions = 0
            num_sessions_since_learning = 0
            done = self.env.done()
            while num_sessions < session_limit and not done:
                # read state and do action selection
                # run selected action, get new state
                # Save info to history, query if done and increment session limit
                # call learning step if number of elapsed episodes is enough
                pass
        pass

    def learn(self):
        pass

    def test(self):
        pass