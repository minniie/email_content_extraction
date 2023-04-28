import random
import math
import json

import torch
from torch.nn.utils.rnn import pad_sequence


DATA_PATH = "/mnt/16tb/minyoung/data/crawler/princeton_emails.json"
SQUAD_LIKE_QUESTIONS = {
    "location": "Where does this event take place?",
    "time": "When does this event take place?",
    "organization": "What is the hosting organization of the event?",
    "title": "What is the title of the event?",
    "guests": "Who are the guests of this event?",
    "required": "What is required for this event?"
}


class PrincetonEmailDataset():

    def __init__(
            self
        ):
        print(">>> loading princeton emails dataset")

    def load(
            self
        ):
        # load preprocessed princeton email dataset from local directory
        with open(DATA_PATH, "r") as f:
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


class PrincetonEmailCollator():

    def __init__(
            self,
            tokenizer,
            data_args
        ):
        self.tokenizer = tokenizer
        self.dataset_format = data_args.dataset_format

    def __call__(self, samples):
        # get body as input, content as output
        bodies, contents = [], []
        for sample in samples:
            
            bodies.append(
                sample["body"] + self.tokenizer.eos_token
            )
            contents.append(
                sample["content"] + self.tokenizer.eos_token
            )

        # comment out for debugging
        # print(bodies)
        # print(contents)
        
        # proper padding for huggingface i/o
        input_ids, labels = [], []
        for b, c in zip(bodies, contents):
            b_ids = self.tokenizer.encode(
                b, max_length=512, truncation=True
            )
            c_ids = self.tokenizer.encode(c)
            input_ids.append(torch.LongTensor(b_ids + c_ids))
            labels.append(torch.LongTensor(len(b_ids)*[-100] + c_ids))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
        # return batch
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }