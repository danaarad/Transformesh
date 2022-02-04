import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class WalkEmbedding(nn.Module):
    def __init__(self, args):
        self.walk_len = args.max_walk_len
        self.emsize = args.emsize
        self.step_features = args.step_features
        self.batch_size = args.batch_size
        super().__init__()
        self.emb = nn.Linear(self.step_features, self.emsize)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emsize))
        # self.positions = nn.Parameter(torch.randn(self.walk_len + 1, self.emsize))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x):
        # print("1. x: ", x.shape)
        x = self.emb(x) * math.sqrt(
            self.emsize)  # [batch_size, seq_length, d_model] project input vectors to d_model dimensional space
        # print("2. self.emb(x): ", x.shape)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_tokens, x), 1)
        # print("3. torch.cat((cls_tokens, x), 1): ", x.shape)
        # print("4. self.positions.shape: ", self.positions.shape)
        # x += self.positions #########
        # print(x.shape)
        return x


class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.args = args
        self.src_mask = None
        self.emb = WalkEmbedding(args)
        self.positional_encoder = LearnablePositionalEncoding(d_model=args.emsize,
                                                              dropout=args.dropout,
                                                              max_len=args.max_walk_len+1)  # +1 for cls token
        encoder_layers = TransformerEncoderLayer(
            d_model=args.emsize,
            nhead=args.nhead,
            dim_feedforward=args.nhid,
            dropout=args.dropout,
            activation=args.activation,
            batch_first=True
        ).to(args.device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.nlayers)
        self.classifier = nn.Linear(args.emsize, args.nclasses)
        self.dropout1 = nn.Dropout(args.dropout)
        self.init_weights()

    def init_weights(self):
        # nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        # print("1. forward - x:", x.shape)
        x = self.emb(x)
        # print("2. forward - self.emb(x):", x.shape)
        ############
        x = self.positional_encoder(x)
        ############
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        ##################
        x = self.dropout1(x)
        ##################
        x = self.classifier(x)
        # print(x.shape)
        return x
