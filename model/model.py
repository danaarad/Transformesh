import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import numpy as np


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, ntokens, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(ntokens, d_model)
        position = torch.arange(0, ntokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.args = args
        self.src_mask = None
        self.emb = nn.Embedding(args.ntokens, args.emsize)
        self.pos_encoder = PositionalEncoding(
            d_model=args.emsize,
            dropout=args.dropout,
            ntokens=args.ntokens
        )
        encoder_layers = TransformerEncoderLayer(
            d_model=args.emsize,
            nhead=args.nhead,
            dim_feedforward=args.nhid,
            dropout=args.dropout,
            batch_first=True
        ).to(args.device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.nlayers)

        self.classifier = nn.Linear(args.nhid, args.nclasses)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        # print(x.shape)
        x = self.emb(x) * math.sqrt(self.args.emsize)
        # print(x.shape)
        x = self.pos_encoder(x)
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        # x = x.mean(dim=1)
        x = self.classifier(x)
        # print(x.shape)
        return x
