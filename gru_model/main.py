# coding: utf-8
import json
import time
import random
import argparse
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import GRUModel
from dataset import Mesh_Dataset


#
# Code adapted from pytorch examples: Word-level language modeling RNN
# https://github.com/pytorch/examples/blob/151944ecaf9ba2c8288ee550143ae7ffdaa90a80/word_language_model/main.py
#


def evaluate_loss(args, model, criterion, dataloader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_index, (data, labels) in enumerate(dataloader):
            data, labels = data.to(args.device), labels.to(args.device)
            hidden_state = model.init_hidden(data.shape[0])

            output, hidden_state = model(data, hidden_state)
            hidden_state = hidden_state.detach()

            output = output.to(args.device)[:, -1, :]
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
            output = output[:, -1, :]
            output = torch.squeeze(output)
            output = torch.softmax(output, dim=1)
            data, labels, output = data.to("cpu"), labels.to("cpu"), output.to("cpu")
            for i in range(len(output)):
                pred = np.argmax(output[i])
                gold = labels[i]
                total_matches += 1 if pred == gold else 0

    return total_matches / len(dataloader.dataset)


def evaluate_acc_by_majority(args, model, data):
    walks_by_mesh, mesh_to_labels = data
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_matches = 0.
    with torch.no_grad():
        for index, (shape_id, data) in enumerate(walks_by_mesh.items()):
            label = mesh_to_labels[shape_id]
            labels = torch.tensor([label]).to(torch.int64)
            labels = labels.tile((args.num_walks, 1))

            data, labels = data.to(args.device), labels.to(args.device)
            hidden_state = model.init_hidden(data.shape[0])
            output, _ = model(data, hidden_state)
            output = output[:, -1, :]
            output = torch.squeeze(output)
            output = torch.softmax(output, dim=1)
            data, labels, output = data.to("cpu"), labels.to("cpu"), output.to("cpu")

            preds_by_class = [0] * args.nclasses
            for i in range(len(output)):
                # pred = np.argmax(output[i])
                # preds_by_class[pred] += 1
                ##############
                for pred in range(output.shape[1]):
                    preds_by_class[pred] = torch.sum(output[:, pred]).cpu().detach().numpy()
                    #################
            majority_class = np.argmax(preds_by_class)
            if majority_class == label:
                total_matches += 1

    return total_matches / len(walks_by_mesh)


def preprocess_for_eval_by_majority(args, data):
    walks_by_mesh = dict()
    mesh_to_labels = dict()
    walk_counters = dict()

    for sample in data.values():
        shape_id = sample["shape_id"]
        if walk_counters.get(shape_id, 0) >= args.num_walks:
            continue

        walk_counters[shape_id] = walk_counters.get(shape_id, 0) + 1

        if shape_id not in walks_by_mesh:
            walks_by_mesh[shape_id] = []

        seq = np.array(sample["dxdydz"])
        seq_len = seq.shape[0]

        # slice to max walk len
        seq = seq[:args.max_walk_len, :]
        # add padding if needed
        if seq_len < args.max_walk_len:
            seq = np.concatenate((seq, np.zeros((args.max_walk_len - seq_len, args.step_features))), axis=0)

        label = sample["shape_label"]
        walks_by_mesh[shape_id].append(seq)
        mesh_to_labels[shape_id] = label

    walks_by_mesh = {k: np.array(v) for k, v in walks_by_mesh.items()}
    walks_by_mesh = {k: torch.from_numpy(v).to(torch.float) for k, v in walks_by_mesh.items()}
    return walks_by_mesh, mesh_to_labels


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
        hidden_state = model.init_hidden(data.shape[0])
        output, hidden_state = model(data, hidden_state)
        hidden_state = hidden_state.detach()

        # output.shape (batch_size, seq_size, nclasses)

        output = output.to(args.device)[:, -1, :]
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
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--max_walk_len', type=int, default=125,
                        help='mex walk length to use')
    parser.add_argument('--num_walks', type=int, default=128,
                        help='number of walks to use per mesh')
    parser.add_argument('--step_features', type=int, default=3,
                        help='number of featuers in the representation of each step. default is 3 for dxdydz')
    parser.add_argument('--nclasses', type=int, default=30,
                        help='number of classification classes')

    parser.add_argument('--train_json', type=str, default="./walks_shrec16_train_dev_walks_128_ratio_05V.json",
                        help='path to file containing data. should be JSON.')
    parser.add_argument('--dev_json', type=str, default="./walks_shrec16_train_dev_walks_128_ratio_05V.json",
                        help='path to file containing data. should be JSON.')
    parser.add_argument('--test_json', type=str, default="./walks_shrec16_test_walks_128_ratio_05V.json",
                        help='path to file containing data. should be JSON.')

    parser.add_argument('--log-interval', type=int, default=200,
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

    args.device = torch.device("cuda:3" if args.cuda else "cpu")
    random.seed(args.seed)

    ###############################################################################
    # Build the model
    ###############################################################################

    model = GRUModel(args).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr
    )

    ###############################################################################
    # Load data
    ###############################################################################

    train_dataset = Mesh_Dataset("train", args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    with open(args.train_json, "rb") as f:
        train_data = json.load(f)

    train_data = {k: v for k, v in train_data.items() if v["split"] == "train"}
    train_data = preprocess_for_eval_by_majority(args, train_data)

    with open(args.dev_json, "rb") as f:
        dev_data = json.load(f)

    dev_data = {k: v for k, v in dev_data.items() if v["split"] == "dev"}
    dev_data = preprocess_for_eval_by_majority(args, dev_data)

    with open(args.test_json, "rb") as f:
        test_data = json.load(f)
    test_data = preprocess_for_eval_by_majority(args, test_data)

    ###############################################################################
    # Training code
    ###############################################################################

    try:
        best_dev_acc = None
        best_epoch = None
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_epoch(args, model, criterion, optimizer, epoch, train_dataloader)

            train_acc = evaluate_acc_by_majority(args, model, train_data)
            dev_acc = evaluate_acc_by_majority(args, model, dev_data)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train accuracy {:5.3f} | dev accuracy {:5.3f}'.format(
                epoch, (time.time() - epoch_start_time), train_acc, dev_acc))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_dev_acc or dev_acc >= best_dev_acc:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_dev_acc = dev_acc
                best_epoch = epoch

                test_acc = evaluate_acc_by_majority(args, model, test_data)
                print('-' * 89)
                print('| best model so far, test accuracy {:5.3f} '.format(test_acc))
                print('-' * 89)

        with open(args.save, 'rb') as f:
            best_model = torch.load(f)
        train_acc = evaluate_acc_by_majority(args, best_model, train_data)
        dev_acc = evaluate_acc_by_majority(args, best_model, dev_data)
        test_acc = evaluate_acc_by_majority(args, best_model, test_data)
        print('-' * 89)
        print('| best model from epoch {}: train accuracy {:5.3f}, dev accuracy {:5.3f}, test accuracy {:5.3f} '.format(
            best_epoch,
            train_acc,
            dev_acc,
            test_acc))
        print('-' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


main()
