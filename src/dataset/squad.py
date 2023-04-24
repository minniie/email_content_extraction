import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


class SquadDataset():

    def __init__(
            self
        ):
        print(">>> loading squad dataset")

    def load(
            self
        ):
        dataset = load_dataset('squad')
        return dataset


class SquadCollator():
    def __init__(
            self,
            tokenizer
        ): 
        self.tokenizer = tokenizer
    
    def __call__(self, samples):
        # get context and question as input, answer as output
        contexts_and_questions, answers = [], []
        for sample in samples:
            contexts_and_questions.append(
                sample["context"] + self.tokenizer.eos_token \
                + sample["question"] + self.tokenizer.eos_token
            )
            answers.append(
                sample["answers"]["text"][0] + self.tokenizer.eos_token
            )

        # comment out for debugging
        # print(contexts_and_questions)
        # print(answers)
        
        # proper padding for huggingface i/o
        input_ids, labels = [], []
        for c_and_q, a in zip(contexts_and_questions, answers):
            c_and_q_ids = self.tokenizer.encode(
                c_and_q, max_length=768, truncation=True
            )
            a_ids = self.tokenizer.encode(a)
            input_ids.append(torch.LongTensor(c_and_q_ids + a_ids))
            labels.append(torch.LongTensor(len(c_and_q_ids)*[-100] + a_ids))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
        # return batch
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }