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
            batch_first=True
        ).to(args.device)

        self.gru2 = GRU(
            input_size=1024,
            hidden_size=1024,
            num_layers=1,
            batch_first=True
        ).to(args.device)

        self.gru3 = GRU(
            input_size=1024,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        ).to(args.device)
        self.classifier = nn.Linear(512, args.nclasses)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def init_hidden(self, layers, batch_size, hidden_dim):
        return torch.zeros(layers, batch_size, hidden_dim).to(self.args.device)

    def forward(self, x):
        # print(x.shape)
        x = self.emb(x)
        # print(x.shape)
        x = self.linear(x)
        h = self.init_hidden(1, x.shape[0], 1024)
        x, h = self.gru1(x, h)
        h = h.detach()

        h = self.init_hidden(1, x.shape[0], 1024)
        x, h = self.gru2(x, h)
        h = h.detach()

        h = self.init_hidden(1, x.shape[0], 512)
        x, h = self.gru3(x, h)
        h = h.detach()

        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x