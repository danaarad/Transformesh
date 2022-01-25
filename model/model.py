import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class WalkEmbedding(nn.Module):
    def __init__(self, args):
        self.walk_len = args.max_walk_len
        self.emsize = args.emsize
        self.step_features = args.step_features
        self.batch_size = args.batch_size
        super().__init__()
        self.emb = nn.Linear(self.step_features, self.emsize)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emsize))
        self.positions = nn.Parameter(torch.randn(self.walk_len + 1, self.emsize))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x):
        # print(x.shape)
        x = self.emb(x)
        # print(x.shape)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_tokens, x), 1)
        # print(x.shape)
        x += self.positions
        # print(x.shape)
        return x


class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.args = args
        self.src_mask = None
        self.emb = WalkEmbedding(args)
        encoder_layers = TransformerEncoderLayer(
            d_model=args.emsize,
            nhead=args.nhead,
            dim_feedforward=args.nhid,
            dropout=args.dropout,
            batch_first=True
        ).to(args.device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.nlayers)
        self.classifier = nn.Linear(args.emsize, args.nclasses)
        self.init_weights()

    def init_weights(self):
        # nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        # print(x.shape)
        x = self.emb(x)
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
