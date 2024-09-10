import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchinfo import summary

import random
import os

from collections import deque
import models.models as models
import envs.base_env as baseenv

class DQN:
    def __init__(self, env: baseenv.BaseEnv, max_history = 50):
        self.env = env
        self.features = env.get_state_shape()
        self.actions = env.get_max_actions()

        # TODO Add a cosine based scaler for epsilon, shift the phase on each epoch.
        self.cosine_scaler = 0.0
        self.history = deque([], max_history)

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda:0'
        self.device = torch.device(DEVICE)
        
        #self.model, self.optim, _ = models.construct_densenetV1(self.features, self.actions, lr=4e-5)
        self.model, self.optim, _ = models.construct_transnet(self.features, self.actions, lr=4e-5)
        print(f'Created a model with {self.features} features and {self.actions} actions.')
        self.model = self.model.to(self.device)
        print(f'Model loaded onto {DEVICE}.')
        print(summary(self.model, input_size=(max_history, self.features)))

    def add_to_history(self, event):
        self.history.append(event)

    # Sample batch_count samples from the replay buffer, may return less depending on buffer count and size.
    def sample_from_history(self, batch_size):
        # Make sure to clamp batch_size to <= current count and max count of buffer.
        batch_size = min(min(batch_size, self.history.maxlen), len(self.history))
        return random.sample(self.history, batch_size)
    
    def predict(self, states):
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        states = states.to(torch.float32)
        states = states.to(self.device)
        self.model = self.model.to(self.device)
        
        with torch.no_grad():
            self.model.train(False)
            outputs = self.model(states)
            return outputs
    
    # Really every call only gives a single state, so this should be cleaned up to represent that
    def get_action(self, states, e=0.0, action_mask=None, action_list=None):
        outputs = self.predict(states)
        if np.random.uniform(0, 1) < e:
            if action_list is not None:
                return [np.random.choice(action_list)]
            return torch.from_numpy(np.random.randint(self.actions, size=len(states)))
        else:
            if action_mask is not None:
                for itr, mask in enumerate(action_mask):
                    if not mask:
                        outputs[0][itr] -= 100
            return torch.argmax(outputs, axis=1).to(torch.int32)
    
    def history_size(self):
        return len(self.history)
    
    # Get the current value of the cosine scaler
    def cosine_scaler_get(self):
        val = (np.cos(self.cosine_scaler) + 1) / 2
        if val <= 0.10:
            val = 0.0
        return val
    
    # Increment the cosine scaler by one step
    def cosine_scaler_increment(self):
        self.cosine_scaler += np.pi / 123
        if self.cosine_scaler >= np.pi * 2:
            self.cosine_scaler -= np.pi * 2

    def cosine_scaler_reset(self, offset):
        self.cosine_scaler = 0 + (np.pi / 123 * offset)
        if self.cosine_scaler >= np.pi * 2:
            self.cosine_scaler -= np.pi * 2

    # Currently getting more exploding gradients at higher gamma values (>0.5).
    # Might need to compute running average reward and lower gamma if it gets too low to combat this.
    # "Nihilistic Lookahead"
    def train(self, gamma = 0.8, num_epochs=1, num_episodes_per_learning_session=10, session_limit=5,
              starting_e = 0.95, min_e = 0.10, e_decay_factor = 0.95):
        #curr_epsilon = starting_e
        best_eval_score = -1000.0
        his_x, his_y = [], []
        for epoch in range(num_epochs):
            self.env.reset_env()
            #print(self.env.state())
            self.cosine_scaler_reset(epoch * 3)
            curr_epsilon = self.cosine_scaler_get()
            loss = 0
            samples = 0
            num_sessions = 0
            num_sessions_since_learning = 0
            done = self.env.is_done()
            rewards = 0
            while num_sessions < session_limit and not done:
                # read state and do action selection
                state = self.env.state()
                valid_actions_mask, valid_actions = self.env.valid_actions()
                #action = self.get_action(state, e=curr_epsilon,
                #                         action_mask=valid_actions_mask, action_list=valid_actions)[0]
                action = self.get_action(state, e=curr_epsilon)[0]
                # run selected action, get new state
                # (res) = reward, pot, dmg
                res = self.env.step(action)
                next_state = self.env.state()
                
                # Save info to history, query if done and increment session count
                event = (state, action, res[0], next_state, self.env.is_done())
                rewards += res[0]
                self.add_to_history(event)
                num_sessions += 1

                # call learning step if number of elapsed episodes is enough
                num_sessions_since_learning += 1
                if(num_sessions_since_learning >= num_episodes_per_learning_session):
                    _loss, _samples = self.learn(gamma)
                    loss += _loss
                    samples += _samples
                    num_sessions_since_learning = 0

                # Adjust epsilon based on cosine scheduling
                self.cosine_scaler_increment()
                curr_epsilon = self.cosine_scaler_get()
            # One more learning session at the very end.
            _loss, _samples = self.learn(gamma)
            loss += _loss
            samples += _samples
            # Call a loop of evaluation then save checkpoint if better
            eval_score = self.eval()
            if eval_score > best_eval_score:
                # TODO have a reference to model type somewhere...
                self.save_checkpoint(f'./checkpoints/_trans_{eval_score:.2f}.pth')
                self.save_checkpoint(f'./checkpoints/_trans_best.pth')
                best_eval_score = eval_score
            # Record History
            his_x.append([len(his_x)+1, len(his_x)+1])
            his_y.append([rewards / samples, eval_score / 50])
            print(f'Epoch {epoch} Loss: {(loss / samples):.3f} E: {curr_epsilon:.3f} G: {gamma:.2f} '+
                  f'Rewards: {rewards:.1f} Eval Rewards: {eval_score:.2f}')
            curr_epsilon = max(curr_epsilon * e_decay_factor, min_e)
        print('Done')
        return his_x, his_y

    def learn(self, gamma = 0.8, sample_count = 128):
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
            updated_val = _reward + gamma * torch.amax(next_pred)
            updated_pred = pred.detach().clone()
            updated_pred[_action] = updated_val
            ys.append(updated_pred.tolist())        
        xs = torch.tensor(xs, dtype=torch.float32, device=self.device)
        ys = torch.tensor(ys, dtype=torch.float32, device=self.device)

        # Fit
        self.optim.zero_grad()
        self.model.train(True)
        #self.model = self.model.to(torch.device('cpu'))
        outputs = self.model(xs)
        loss = loss_func(outputs, ys)
        loss.backward()
        # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        # To help mitigate the exploding gradient. TODO find optimal value
        #  - 10 is probably too high
        #if loss.item() / sample_count > 4000:
        #nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

        self.optim.step()
        #print(loss.item())
        #if loss.item() > 5000:
            #print(self.predict())
        return loss.item(), sample_count
    
    def eval(self, num_steps = 50):
        self.env.reset_env()
        rewards = 0.0
        for _ in range(num_steps):
            if self.env.is_done():
                break
            state = self.env.state()
            action = self.get_action(state, e=0.0)[0]
            reward, _, _ = self.env.step(action)
            rewards += reward
        return rewards

    def test(self, num_steps = 10):
        self.env.reset_env()
        actions_taken = []
        print(self.env.state())
        for iter in range(num_steps):
            #print(self.env.state())
            done = self.env.is_done()
            if done:
                break
            # read state and do action selection
            state = self.env.state()
            #print(self.predict(state))
            action = self.get_action(state, e=0.0)[0]
            actions_taken.append(action)
            # run selected action, get new state
            # (res) = reward, pot, dmg
            res = self.env.step(action, _verbose = True)
            print(res)

    # Checkpointing code
    def save_checkpoint(self, path):
        save = {'model': self.model.state_dict(),
                'opti': self.optim.state_dict()}
        torch.save(save, path)

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            self.optim.load_state_dict(checkpoint['opti'])
        else:
            print('File not found.')