import json
import numpy as np

import torch
from torch.utils.data import Dataset


class Mesh_Dataset(Dataset):
    def __init__(self, split, args):
        self.split = split
        self.filename = self.get_filename_from_split(args)
        self.num_walks = args.num_walks
        self.max_walk_len = args.max_walk_len
        self.step_features = args.step_features
        self.features_type = args.features_type
        self.nclasses = args.nclasses
        self.seqs, self.labels = self.process_data()

    def get_filename_from_split(self, args):
        if self.split == "train":
            return args.train_json
        elif self.split == "dev":
            return args.dev_json
        elif self.split == "test":
            return args.test_json
        else:
            raise Exception(f"Unknown split: {self.split}")

    def process_data(self):
        with open(self.filename, "rb") as f:
            data = json.load(f)

        walk_counters = dict()
        seqs = []
        labels = []

        for sample in data.values():
            if sample["split"] == self.split:
                shape_id = sample["shape_id"]
                if walk_counters.get(shape_id, 0) >= self.num_walks:
                    continue

                walk_counters[shape_id] = walk_counters.get(shape_id, 0) + 1
                seq = np.array(sample[self.features_type])
                seq_len = seq.shape[0]

                # slice to max walk len
                seq = seq[:self.max_walk_len, :]
                # seq = np.concatenate((np.zeros((1, 3)), seq), axis=0)

                # add padding
                if seq_len < self.max_walk_len:
                    seq = np.concatenate((seq, np.zeros((self.max_walk_len - seq_len, self.step_features))), axis=0)

                # rescale numbers
                # seq = seq * 1000000

                # seq = seq.flatten()
                # seq = np.expand_dims(seq, axis=1)
                label = sample["shape_label"]

                seqs.append(seq)
                labels.append(label)

        seqs = np.array(seqs)
        labels = np.array(labels)

        print(len(set(labels)))
        print(sorted(list(set(labels))))
        print(seqs.shape)
        print(labels.shape)
        seqs_tensor = torch.from_numpy(seqs).to(torch.float)
        labels_tensor = torch.tensor(labels).to(torch.int64)
        return seqs_tensor, labels_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]
