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
    def __init__(self, env: baseenv.BaseEnv, epsilon = 1.0, max_history = 50):
        self.env = env
        self.features = env.get_state_shape()
        self.actions = env.get_max_actions()
        self.epsilon = epsilon
        self.history = deque([], max_history)

        self.model, self.optim, _ = models.construct_densenet(self.features, self.actions)
        self.model = self.model.to(torch.device('cpu'))
        print(summary(self.model, input_size=(8, self.features)))

    def add_to_history(self, event):
        self.history.append(event)

    def sample_from_history(self, batch_size):
        batch_size = min(min(batch_size, self.history.maxlen), len(self.history))
        return random.sample(self.history, batch_size)
    
    def predict(self, states):
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
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
            return torch.from_numpy(np.random.randint(self.actions, size=len(states)))
        else:
            return torch.argmax(outputs, axis=1).to(torch.int32)
    
    def history_size(self):
        return len(self.history)

    def train(self, num_epochs=1, num_episodes_per_learning_session=10, session_limit=5):
        for epoch in range(num_epochs):
            self.env.reset_env()
            #print(self.env.state())
            loss = 0
            samples = 0
            num_sessions = 0
            num_sessions_since_learning = 0
            done = self.env.is_done()
            while num_sessions < session_limit and not done:
                # read state and do action selection
                state = self.env.state()
                action = self.get_action(state, e=0.70)[0]
                
                # run selected action, get new state
                # (res) = reward, pot, dmg
                res = self.env.step(action)
                next_state = self.env.state()
                #print(res)
                #print(self.env.state())
                
                # Save info to history, query if done and increment session count
                event = (state, action, res[0], next_state, self.env.is_done())
                self.add_to_history(event)
                num_sessions += 1
                # call learning step if number of elapsed episodes is enough
                num_sessions_since_learning += 1
                if(num_sessions_since_learning >= num_episodes_per_learning_session):
                    _loss, _samples = self.learn()
                    loss += _loss
                    samples += _samples
                    num_sessions_since_learning = 0
            print(f'Epoch {epoch} Loss: {(loss / samples):.3f}')
        print('Done')

    def learn(self, gamma = 0.5, sample_count = 128):
        # OFF-POLICY Approach
        #loss_func = nn.L1Loss()
        loss_func = nn.MSELoss()

        # Sample from history
        samples = self.sample_from_history(sample_count)
        xs, ys = [], []
        for sample in samples:
            _state, _action, _reward, _next_state, _done = sample
            # Re-predict the values
            pred = self.predict(_state)[0]
            xs.append(_state.tolist())
            next_pred = self.predict(_next_state)[0]
            # Update the values at the selected action
            # reward + gamma * (np.amax(predict(next_state)))
            #print(f'reward: {_reward}')
            updated_val = _reward + gamma * torch.amax(next_pred)
            updated_pred = pred.detach().clone()
            updated_pred[_action] = updated_val
            ys.append(updated_pred.tolist())
            
        xs = torch.tensor(xs, dtype=torch.float32, device=torch.device('cpu'))
        ys = torch.tensor(ys, dtype=torch.float32, device=torch.device('cpu'))

        # Fit
        self.optim.zero_grad()
        self.model.train(True)
        #self.model = self.model.to(torch.device('cpu'))
        outputs = self.model(xs)
        loss = loss_func(outputs, ys)
        loss.backward()
        # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        
        self.optim.step()
        #print(loss.item())
        #if loss.item() > 5000:
            #print(self.predict())
        return loss.item(), sample_count

    def test(self, num_steps = 10):
        self.env.reset_env()
        for iter in range(num_steps):
            print(self.env.state())
            done = self.env.is_done()
            if done:
                break
            # read state and do action selection
            state = self.env.state()
            #print(self.predict(state))
            action = self.get_action(state, e=0.0)[0]
            
            # run selected action, get new state
            # (res) = reward, pot, dmg
            res = self.env.step(action, _verbose = True)
            print(res)
            next_state = self.env.state()
            #print(res)
            #print(self.env.state())