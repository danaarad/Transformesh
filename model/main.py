# coding: utf-8
import os
import math
import time
import random
import argparse
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertTokenizer

from model import TransformerModel
from dataset import IMDBDataset


#
# Code adapted from pytorch examples: Word-level language modeling RNN
# https://github.com/pytorch/examples/blob/151944ecaf9ba2c8288ee550143ae7ffdaa90a80/word_language_model/main.py
#

def get_batch(train_data, train_labels, batch_size, batch_index):
    batch_data = train_data[batch_size * batch_index: batch_size * batch_index + batch_size]
    batch_labels = train_labels[batch_size * batch_index: batch_size * batch_index + batch_size]
    return batch_data, batch_labels


def evaluate_loss(args, model, criterion, dataloader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_index, (data, labels) in enumerate(dataloader):
            data, labels = data.to(args.device), labels.to(args.device)
            output = model(data)
            output = output.to(args.device)[:, 0, :]
            output = torch.squeeze(output)
            total_loss += criterion(output, labels).item()
    return total_loss / len(dataloader)


def evaluate_acc(args, model, dataloader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_matches = 0.
    with torch.no_grad():
        for batch_index, (data, labels) in enumerate(dataloader):
            data, labels = data.to(args.device), labels.to(args.device)
            output = model(data)
            output = output[:, 0, :]
            output = torch.squeeze(output)
            output = torch.softmax(output, dim=1)
            data, labels, output = data.to("cpu"), labels.to("cpu"), output.to("cpu")
            for i in range(len(output)):
                pred = np.argmax(output[i])
                gold = labels[i]
                total_matches += 1 if pred == gold else 0

    return total_matches / len(dataloader.dataset)


def train_epoch(args, model, criterion, optimizer, epoch, dataloader):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch_index, (data, labels) in enumerate(dataloader):
        data, labels = data.to(args.device), labels.to(args.device)
        # data.shape (batch_size, seq_size)
        # labels.shape (nclasses)

        optimizer.zero_grad()
        output = model(data)
        # output.shape (batch_size, seq_size, nclasses)

        output = output.to(args.device)[:, 0, :]
        output = torch.squeeze(output)
        # output.shape (batch_size, nclasses)


        # The output is expected to contain scores for each class.
        # output has to be a 2D Tensor of size (minibatch, C).
        # This criterion expects a class index (0 to C-1) as the target
        # for each value of a 1D tensor of size minibatch
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_index % args.log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / batch_index
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} '.format(epoch, batch_index, len(dataloader), args.lr,
                                         elapsed * 1000 / args.log_interval, cur_loss))
            start_time = time.time()


def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Implementation')
    parser.add_argument('--data', type=str, default='./data/imdb',
                        help='location of the data corpus')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')

    parser.add_argument('--emsize', type=int, default=128,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the transformer model')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='initial momentum')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='batch size')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='mex sequence length size')

    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')

    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    args.device = torch.device("cuda:2" if args.cuda else "cpu")
    random.seed(args.seed)

    ###############################################################################
    # Build the model
    ###############################################################################

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args.ntokens = len(tokenizer.vocab)
    args.nclasses = 2
    model = TransformerModel(args).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr
    )

    ###############################################################################
    # Load data
    ###############################################################################

    raw_train_data = load_dataset('imdb', split='train')
    raw_test_data = load_dataset('imdb', split='test')

    raw_train_indices = random.sample(range(len(raw_train_data)), k=int(len(raw_train_data) * 0.9))
    raw_dev_data = [sample for i, sample in enumerate(raw_train_data) if i not in raw_train_indices]
    raw_train_data = [sample for i, sample in enumerate(raw_train_data) if i in raw_train_indices]

    train_data = IMDBDataset(raw_train_data, args.max_seq_len, args.nclasses)
    dev_data = IMDBDataset(raw_dev_data, args.max_seq_len, args.nclasses)
    test_data = IMDBDataset(raw_test_data, args.max_seq_len, args.nclasses)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    ###############################################################################
    # Training code
    ###############################################################################

    try:
        best_dev_loss = None
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_epoch(args, model, criterion, optimizer, epoch, train_dataloader)

            train_acc = evaluate_acc(args, model, train_dataloader)

            dev_loss = evaluate_loss(args, model, criterion, dev_dataloader)
            dev_acc = evaluate_acc(args, model, dev_dataloader)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train accuracy {:5.3f} | dev loss {:5.2f} |'
                  ' dev accuracy  {:5.3f}'.format(
                epoch, (time.time() - epoch_start_time), train_acc, dev_loss, dev_acc))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_dev_loss or dev_loss < best_dev_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_dev_loss = dev_loss
                # else:
                #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
                #     lr /= 4.0

                # Run on test data.
                # test_loss = evaluate_loss(args, model, criterion, test_dataloader)
                # test_acc = evaluate_acc(args, model, test_dataloader)
                # print('=' * 89)
                # print('| test loss {:5.2f} | test accuracy {}'.format(test_loss, test_acc))
                # print('=' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


main()
