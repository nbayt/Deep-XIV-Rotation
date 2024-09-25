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
    def __init__(self, _env: baseenv.BaseEnv, _max_history = 50, _batch_size = 196):
        self.env = _env
        self.features = _env.get_state_shape()
        self.actions = _env.get_max_actions()

        # Cosine based scaler for epsilon, shifts the phase on each epoch.
        self.cosine_scaler = 0.0
        self.cosine_scaler_div = 180 #123
        self.epoch_offset = 0
        self.history = deque([], _max_history)
        self.batch_size = _batch_size

        self.training_history_x = []
        self.training_history_y = []

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda:0'
        self.device = torch.device(DEVICE)

        self.lr = 6.0e-5 # 4.5?
        
        self.model, self.optim, self.scheduler, self.model_name = models.construct_densenetV1(self.features, self.actions, lr=self.lr)
        #self.model, self.optim, self.scheduler, self.model_name = models.construct_transnet(self.features, self.actions, lr=self.lr)
        print(f'Created model {self.model_name} with {self.features} features and {self.actions} actions.')

        self.model = self.model.to(self.device)
        print(f'Model loaded onto {DEVICE}.')

        print(summary(self.model, input_size=(self.batch_size, self.features)))

    def add_to_history(self, event):
        """Adds an event to the history buffer. Expects an input tuple of: (state, action, reward, next_state, is_done)."""
        self.history.append(event)

    # Sample batch_count samples from the replay buffer, may return less depending on buffer count and size.
    def sample_from_history(self, batch_size):
        """Randomly samples batch_size samples from the history buffer. This amount will be constrained by
        the current count of the buffer, as well as the maximum size of the buffer."""
        # Make sure to clamp batch_size to <= current count and max count of buffer.
        batch_size = min(min(batch_size, self.history.maxlen), self.history_size())
        return random.sample(self.history, batch_size)
    
    def predict(self, states: torch.Tensor):
        """Gives the model's prediction for a set of input states.\n
        Will expand input to dimension of size 2 if necessary, result will also have a dimension of 2 as well."""
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        states = states.to(self.device, dtype=torch.float32)
        self.model = self.model.to(self.device)
        
        with torch.no_grad():
            self.model.train(False)
            outputs = self.model(states)
            return outputs
        
    # Softmax selection
    def get_action_softmax(self, state, e=0.0):
        """Returns an action following softmax weighting with probability 1-e."""
        outputs = self.predict(state)
        if np.random.uniform(0, 1) < e:
            return torch.from_numpy(np.random.randint(self.actions, size = 1))[0]
        else:
            soft_max = torch.nn.Softmax(dim=1)
            _outs = soft_max(outputs)
            return torch.multinomial(_outs, 1, replacement=True)[0][0]
    
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
        """Returns the number of events stored in the history buffer."""
        return len(self.history)
    
    # Get the current value of the cosine scaler.
    def cosine_scaler_get(self):
        """Gets the current value of the cosine scaler, in the range of [0.0, 0.25]. Values below 0.05 are clamped to 0."""
        val = (np.cos(self.cosine_scaler) + 1) / 2
        val = val * 0.25
        if val <= 0.05:
            val = 0.0
        return val
    
    # Increment the cosine scaler by one step.
    def cosine_scaler_increment(self):
        """Increments the cosine scaler by one step."""
        self.cosine_scaler += np.pi / self.cosine_scaler_div
        if self.cosine_scaler >= np.pi * 2:
            self.cosine_scaler -= np.pi * 2

    # Resets the cosine scaler, also increments the epoch offset of the agent.
    def cosine_scaler_reset(self, _increment=True):
        """Resets the cosine scaler to it's inital value, along with shifting the phase based on the number
        of elapsed epochs.\n
        By default, the epoch_offset will also be incremented."""
        self.cosine_scaler = 0 + (np.pi / self.cosine_scaler_div * self.epoch_offset)
        if self.cosine_scaler >= np.pi * 2:
            self.cosine_scaler -= np.pi * 2
        if _increment:
            self.epoch_offset += 1

    # "Nihilistic Lookahead" Appears to mostly have been solved by lowering overall lr.
    def train(self, gamma = 0.8, num_epochs=1, num_episodes_per_learning_session=10, session_limit=5):
        best_eval_score = -1000.0
        for epoch in range(num_epochs):
            # TODO, vary sks from a preset sample
            # 2.12, 2.11, 2.10, 2.09, 2.08
            chosen_sks = np.random.choice([420, 528, 582, 768, 822])
            self.env.reset_env(chosen_sks)
            self.cosine_scaler_reset()

            initial_epsilon = self.cosine_scaler_get()
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
                #valid_actions_mask, valid_actions = self.env.valid_actions()
                #action = self.get_action(state, e=curr_epsilon,
                #                         action_mask=valid_actions_mask, action_list=valid_actions)[0]
                action = self.get_action_softmax(state, e=curr_epsilon)
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
                    _loss, _samples = self.learn(gamma, sample_count=self.batch_size)
                    loss += _loss
                    samples += _samples
                    num_sessions_since_learning = 0

                # Adjust epsilon based on cosine scheduling
                self.cosine_scaler_increment()
                curr_epsilon = self.cosine_scaler_get()

            # One more learning session at the very end.
            _loss, _samples = self.learn(gamma, sample_count=self.batch_size)
            loss += _loss

            samples += _samples
            # Call a loop of evaluation
            eval_score = self.eval()

            lr = self.lr if self.scheduler is None else self.scheduler.get_last_lr()[0]

            # Record History
            self.training_history_x.append([len(self.training_history_x)+1, len(self.training_history_x)+1])
            self.training_history_y.append([rewards / samples, eval_score / 50])

            print(f'Epoch {epoch} Loss: {(loss / samples):.2e} E_0: {initial_epsilon:.2f} E_1: {curr_epsilon:.2f} G: {gamma:.2f} '+
                  f'Rewards: {rewards:.1f} Eval Rewards: {eval_score:.2f}, LR: {lr:.1e} SKS: {self.env.sks}')
            # Step scheduler if possible.
            if self.scheduler is not None:
                self.scheduler.step()
            # Save checkpoints.
            if eval_score > best_eval_score:
                self.save_checkpoint(f'./checkpoints/_{self.model_name}_{eval_score:.2f}_{self.env.sks}.pth')
                self.save_checkpoint(f'./checkpoints/_{self.model_name}_best.pth')
                best_eval_score = eval_score
            self.save_checkpoint(f'./checkpoints/_{self.model_name}_last.pth')
        print('Done')

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
        if loss.item() / len(samples) > 5.0:
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            #print('Clipped grad.')

        self.optim.step()

        return loss.item(), len(samples)

    # TODO merge these two for cleanup.
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

    def test(self, num_steps = 10, _sks=None):
        self.env.reset_env(_sks)
        actions_taken = []
        total_rewards = 0
        print(self.env.state())
        for _ in range(num_steps):
            #print(self.env.state())
            done = self.env.is_done()
            if done:
                break
            # read state and do action selection
            state = self.env.state()

            action = self.get_action(state, e=0.0)[0]
            actions_taken.append(action)
            # run selected action, get new state
            # (res) = reward, pot, dmg
            res = self.env.step(action, _verbose = True)
            print(f'({res[0]:.2f}, {res[1]:.0f}, {res[2]:.2f})')
            total_rewards += res[0]
        return total_rewards

    # Checkpointing code
    def save_checkpoint(self, path):
        # TODO, handle cosine state as well.
        save = {'model': self.model.state_dict(),
                'opti': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'history_x': self.training_history_x,
                'history_y': self.training_history_y,
                'epoch_offset': self.epoch_offset}
        torch.save(save, path)

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            self.optim.load_state_dict(checkpoint['opti'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.training_history_x = checkpoint['history_x']
            self.training_history_y = checkpoint['history_y']
            self.epoch_offset = checkpoint['epoch_offset']
        else:
            print('File not found.')