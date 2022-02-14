import torch
import torch.nn as nn
from torch.nn import GRU


class GRUModel(nn.Module):
    def __init__(self, args):
        super(GRUModel, self).__init__()
        self.args = args
        self.src_mask = None
        self.emb = nn.Linear(self.args.step_features, self.args.emsize)
        self.linear = nn.Linear(self.args.emsize, 256)
        self.gru1 = GRU(
            input_size=256,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
            dropout=self.args.dropout
        ).to(args.device)

        self.gru2 = GRU(
            input_size=1024,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
            dropout=self.args.dropout
        ).to(args.device)

        self.gru3 = GRU(
            input_size=1024,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            dropout=self.args.dropout
        ).to(args.device)
        self.classifier = nn.Linear(512, args.nclasses)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def init_hidden(self, batch_size=None):
        if not batch_size:
            batch_size = self.args.batch_size
        return torch.zeros(self.args.nlayers, batch_size, 1024).to(self.args.device)

    def forward(self, x, h):
        # print(x.shape)
        x = self.emb(x)
        # print(x.shape)
        x = self.linear(x)
        x, h = self.gru(x, h)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x, h
