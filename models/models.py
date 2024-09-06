import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DenseNet(nn.Module):
    def __init__(self, num_features, num_actions):
        super(DenseNet, self).__init__()

        self.fc1 = nn.Linear(num_features, 2048)
        self.fc2 = nn.Linear(2048, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
def construct_densenet(num_features, num_actions):
    model = DenseNet(num_features, num_actions)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # no scheduler
    return model, optimizer, None