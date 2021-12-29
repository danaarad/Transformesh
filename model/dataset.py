import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer



class IMDBDataset(Dataset):
    def __init__(self, raw_data, max_len, nclasses):
        self.max_len = max_len
        self.nclasses = nclasses
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.raw_data = raw_data
        self.reviews, self.labels = self.process_data(raw_data)

    def tokenize(self, data):
        eos_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        init_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)

        tokenized_data = np.zeros((len(data), self.max_len))
        for i, text in enumerate(data):
            tokens = self.tokenizer.tokenize(text)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)

            if len(ids) < self.max_len - 1:
                tokenized_input = [init_token_idx] + ids + [eos_token_idx] * (self.max_len - len(ids) - 1)
            else:
                tokenized_input = [init_token_idx] + ids[:self.max_len - 2] + [eos_token_idx]

            tokenized_array = np.array(tokenized_input, dtype=object)
            tokenized_data[i] = tokenized_array

        return tokenized_data

    def process_data(self, raw_data):
        reviews = [s["text"] for s in raw_data]
        labels = [s["label"] for s in raw_data]

        reviews = self.tokenize(reviews)
        full_labels = np.zeros((len(labels), self.nclasses))
        for i in range(len(labels)):
            full_labels[i][labels[i]] = 1

        reviews_tensor = torch.from_numpy(reviews).to(torch.int64)
        labels_tensor = torch.from_numpy(full_labels).to(torch.int64)
        return reviews_tensor, labels_tensor

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]
