import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DenseNetV1(nn.Module):
    def __init__(self, _num_features, _num_actions):
        super(DenseNetV1, self).__init__()

        self.sequence = nn.Sequential(
            nn.Linear(_num_features, 4096),
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

            nn.Linear(1024, _num_actions)
        )

    def forward(self, x):
        x = self.sequence(x)
        return x
    
def construct_densenetV1(num_features, num_actions, lr=0.001):
    model = DenseNetV1(num_features, num_actions)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # no scheduler
    return model, optimizer, None

class TransformerNet(nn.Module):
    def __init__(self, _num_features, _num_actions, _hidden_dim=512, _hidden_dim_mult=4):
        super(TransformerNet, self).__init__()
        self.hidden_dim = _hidden_dim
        self.hidden_dim_mult = _hidden_dim_mult
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        self.tokenizer = nn.Sequential(
            nn.Linear(1, self.hidden_dim * self.hidden_dim_mult),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.hidden_dim_mult, self.hidden_dim),
            nn.BatchNorm1d(_num_features),
            nn.ReLU(),
            nn.Dropout(p=0.25),
        )

        self.encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, nhead=16,
                                                        dim_feedforward=self.hidden_dim * self.hidden_dim_mult, dropout=0.25,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 4)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, _num_actions),
        )

    def forward(self, x: torch.Tensor):
        b, f = x.shape
        x = x.reshape(b, f, 1)
        tokens = self.tokenizer(x)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        tokens_out = self.encoder(tokens)
        token = tokens_out[:, 0]
        x = self.classifier(token)
        return x

def construct_transnet(num_features, num_actions, lr=0.001):
    model = TransformerNet(num_features, num_actions)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # no scheduler
    return model, optimizer, None
