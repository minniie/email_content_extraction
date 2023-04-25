import random
import math
import json

import torch
from torch.nn.utils.rnn import pad_sequence


class EnronDataset():

    def __init__(
            self
        ):
        print(">>> loading enron dataset")

    def load(
            self
        ):
        # load preprocessed enron dataset from local directory
        data_path = "/mnt/16tb/minyoung/data/enron/enron.json"
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # train and valid split 
        split = math.floor(len(data) / 5)
        random.shuffle(data)
        train_data = data[split:]
        valid_data = data[:split]
        return {
            "train": train_data,
            "validation": valid_data
        }


class EnronCollator():

    def __init__(
            self,
            tokenizer
        ): 
        self.tokenizer = tokenizer

    def __call__(self, samples):
        # get content as input, subject as output
        contents, subjects = [], []
        for sample in samples:
            contents.append(
                sample["content"] + self.tokenizer.eos_token
            )
            subjects.append(
                sample["subject"] + self.tokenizer.eos_token
            )

        # comment out for debugging
        # print(contents)
        # print(subjects)
        
        # proper padding for huggingface i/o
        input_ids, labels = [], []
        for c, s in zip(contents, subjects):
            c_ids = self.tokenizer.encode(
                c, max_length=512, truncation=True
            )
            s_ids = self.tokenizer.encode(s)
            input_ids.append(torch.LongTensor(c_ids + s_ids))
            labels.append(torch.LongTensor(len(c_ids)*[-100] + s_ids))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
        # return batch
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }