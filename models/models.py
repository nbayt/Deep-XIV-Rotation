import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DenseNet(nn.Module):
    def __init__(self, num_features, num_actions):
        super(DenseNet, self).__init__()

        self.sequence = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(),

            #nn.Linear(4096, 4096),
            #nn.ReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            #nn.Linear(2048, 2048),
            #nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(2048, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, num_actions)
        )

    def forward(self, x):
        x = self.sequence(x)
        return x
    
def construct_densenet(num_features, num_actions, lr=0.001):
    model = DenseNet(num_features, num_actions)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # no scheduler
    return model, optimizer, None