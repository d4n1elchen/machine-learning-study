### First, tokenize the input
import os
import torch
import pandas as pd
import numpy as np

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from torch.nn import KLDivLoss, NLLLoss, LogSoftmax, Softmax
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from tqdm import tqdm, trange

OUTPUT_DIR = "saved_model"
EPOCH = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

num_labels = 7

data = pd.read_csv("data.csv")

logsoftmax = LogSoftmax(dim=-1)
softmax = Softmax(dim=-1)

# Load a trained model and vocabulary that you have fine-tuned
model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR)

model.to(device)

def to_token(sentence, max_seq_length=256):
    # Tokenized input
    tokenized_text = tokenizer.tokenize(sentence)

    if len(tokenized_text) > max_seq_length:
        tokenized_text = tokenized_text[:max_seq_length-2]

    tokens = ["[CLS]"] + tokenized_text + ["[SEP]"]

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

    padding = [0] * (max_seq_length - len(indexed_tokens))
    tokens = indexed_tokens + padding
    segments = [0] * len(indexed_tokens) + padding
    masks = [1] * len(indexed_tokens) + padding

    return tokens, segments, masks

def cross_entropy(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

all_tokens = []
all_segments = []
all_masks = []
all_scores = []
for idx, row in data.iterrows():
    sentence = row[0]
    score = torch.tensor(row[1:].astype(np.float))
    score_p = softmax(score)

    tokens, segments, masks = to_token(sentence)

    all_tokens.append(tokens)
    all_segments.append(segments)
    all_masks.append(masks)
    all_scores.append(score_p.tolist())

all_tokens_tensor = torch.tensor(all_tokens)
all_segments_tensor = torch.tensor(all_segments)
all_masks_tensor = torch.tensor(all_masks)
all_scores_tensor = torch.tensor(all_scores)

## Evaluation
model.eval()
# Get random 10 sentences
rand_idx = np.random.choice(len(data), 10)
tokens_ids = all_tokens_tensor[rand_idx]
segments_ids = all_segments_tensor[rand_idx]
masks = all_masks_tensor[rand_idx]
scores = all_scores_tensor[rand_idx]

tokens_ids = tokens_ids.to(device)
segments_ids = segments_ids.to(device)
masks = masks.to(device)
scores = scores.to(device)

with torch.no_grad():
    logits = model(tokens_ids, segments_ids, attention_mask=masks)
    prediction = softmax(logits)

    print(prediction)
    print(scores)
