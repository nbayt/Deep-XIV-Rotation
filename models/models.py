import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DenseNetV3(nn.Module):
    def __init__(self, _num_features, _num_actions):
        super(DenseNetV3, self).__init__()

        self.sequence = nn.Sequential(
            nn.Linear(_num_features, 4096),
            nn.SELU(),

            nn.Linear(4096, 4096),
            nn.Tanh(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Dropout(p=0.25),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Tanh(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Tanh(),

            nn.Linear(1024, 512),
            nn.SELU(),

            nn.Linear(512, 512),
            nn.SELU(),

            nn.Linear(512, _num_actions)
        )

    def forward(self, x):
        x = self.sequence(x)
        return x
    
def construct_densenetV3(num_features, num_actions, lr=0.001):
    model = DenseNetV3(num_features, num_actions)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler_lr_0 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=75)
    scheduler_lr_1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.75, total_iters=200)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_lr_0, scheduler_lr_1], milestones=[1000])
    return model, optimizer, scheduler, 'dense_v1'

class DenseNetV2(nn.Module):
    def __init__(self, _num_features, _num_actions):
        super(DenseNetV2, self).__init__()

        self.sequence = nn.Sequential(
            nn.Linear(_num_features, 4096),
            nn.ReLU(),

            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Dropout(p=0.25),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, _num_actions)
        )

    def forward(self, x):
        x = self.sequence(x)
        return x
    
def construct_densenetV2(num_features, num_actions, lr=0.001):
    model = DenseNetV2(num_features, num_actions)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler_lr_0 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=100)
    scheduler_lr_1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.75, total_iters=200)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_lr_0, scheduler_lr_1], milestones=[1000])
    return model, optimizer, scheduler, 'dense_v1'

class DenseNetV1(nn.Module):
    def __init__(self, _num_features, _num_actions):
        super(DenseNetV1, self).__init__()

        self.sequence = nn.Sequential(
            nn.Linear(_num_features, 4096),
            nn.ReLU(),

            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Dropout(p=0.25),

            nn.Linear(2048, 2048),
            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(2048, 2048),
            nn.ReLU(),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
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
    scheduler_lr_0 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=100)
    scheduler_lr_1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.75, total_iters=200)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_lr_0, scheduler_lr_1], milestones=[1000])
    return model, optimizer, scheduler, 'dense_v1'

class DenseNetV0(nn.Module):
    def __init__(self, _num_features, _num_actions):
        super(DenseNetV0, self).__init__()

        self.sequence = nn.Sequential(
            nn.Linear(_num_features, 4096),
            nn.ReLU(),

            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Dropout(p=0.25),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, _num_actions)
        )

    def forward(self, x):
        x = self.sequence(x)
        return x
def construct_densenetV0(num_features, num_actions, lr=0.001):
    model = DenseNetV0(num_features, num_actions)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler_lr_0 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, end_factor=1.0, total_iters=25)
    scheduler_lr_1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=50)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_lr_0, scheduler_lr_1], milestones=[200])
    return model, optimizer, scheduler, 'dense_v0'

class TransformerNetv0(nn.Module):
    def __init__(self, _num_features, _num_actions, _hidden_dim=512, _hidden_dim_mult=4):
        super(TransformerNetv0, self).__init__()
        self.hidden_dim = _hidden_dim
        self.hidden_dim_mult = _hidden_dim_mult
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        self.tokenizer = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(_num_features),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim * self.hidden_dim_mult),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.hidden_dim_mult, self.hidden_dim),
            #nn.BatchNorm1d(_num_features),
            nn.ReLU(),
            nn.Dropout(p=0.25),
        )

        self.encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, nhead=16,
                                                        dim_feedforward=self.hidden_dim * self.hidden_dim_mult, dropout=0.25,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 6)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
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

def construct_transnetv0(num_features, num_actions, lr=0.001):
    model = TransformerNetv0(num_features, num_actions)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # no scheduler
    return model, optimizer, None, 'transnet_v0'

class TransformerNetv1(nn.Module):
    def __init__(self, _num_features, _num_actions, _hidden_dim=512, _hidden_dim_mult=4, _his_len=4):
        super(TransformerNetv1, self).__init__()
        self.hidden_dim = _hidden_dim
        self.hidden_dim_mult = _hidden_dim_mult
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        self.pos_embed = nn.Parameter(torch.tensor(self.get_positional_embeddings(_his_len + 1, self.hidden_dim)))
        self.pos_embed.requires_grad = False

        self.tokenizer = nn.Sequential(
            nn.Linear(_num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(_his_len),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim * self.hidden_dim_mult),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.hidden_dim_mult, self.hidden_dim),
            #nn.BatchNorm1d(_num_features),
            nn.SELU(),
            nn.Dropout(p=0.25),
        )

        self.encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, nhead=16,
                                                        dim_feedforward=self.hidden_dim * self.hidden_dim_mult, dropout=0.25,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 6)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, _num_actions),
        )

    def get_positional_embeddings(self, sequence_length, d):
        res = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                res[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return res

    def forward(self, x: torch.Tensor):
        b, s, f = x.shape
        #x = x.reshape(b, f, 1)
        #print(x.shape)
        tokens = self.tokenizer(x)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        pos_embed = self.pos_embed.repeat(b, 1, 1)
        tokens = tokens + pos_embed
        tokens_out = self.encoder(tokens)
        token = tokens_out[:, 0]
        x = self.classifier(token)
        return x

def construct_transnetv1(num_features, num_actions, lr=0.001, his_len=4):
    model = TransformerNetv1(num_features, num_actions, _his_len=his_len)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler_lr_0 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=50)
    scheduler_lr_1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.75, total_iters=20)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_lr_0, scheduler_lr_1], milestones=[300])

    return model, optimizer, scheduler, 'transnet_v1'
