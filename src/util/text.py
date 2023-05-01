from typing import List
from collections import Counter

import torch
import numpy as np


def clean_text(text):
    text = text.replace(u'\xa0', u' ')
    text = text.replace(u'\u200b', u'')

    return text


def clean_decode(batch, tokenizer):
    batch = torch.where(batch < 0, tokenizer.pad_token_id, batch)
    text = tokenizer.batch_decode(np.expand_dims(batch, axis=-1), skip_special_tokens=True)
    # text = [t.lower().strip() for t in text]
    text =  list(filter(None, text))
    text = ['NONE'] if not text else text
    # print(text)

    return text


"""Modified from https://rajpurkar.github.io/SQuAD-explorer/"""
def compute_f1(preds_text, labels_text):
    f1_list = []
    for pred, label in zip(preds_text, labels_text):
        overlap = Counter(pred) & Counter(label)
        n_overlap = sum(overlap.values())
        if n_overlap == 0:
            f1_each = 0
        else:
            precision = 1.0 * n_overlap / len(pred)
            recall = 1.0 * n_overlap / len(label)
            f1_each = (2 * precision * recall) / (precision + recall)
        f1_list.append(f1_each)
    f1 = sum(f1_list)/len(f1_list)

    return f1