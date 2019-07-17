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
EPOCH = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

num_labels = 7

tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-chinese', do_basic_tokenize=False)

data = pd.read_csv("data.csv")

logsoftmax = LogSoftmax(dim=-1)
softmax = Softmax(dim=-1)

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

train_data = TensorDataset(all_tokens_tensor, all_segments_tensor, all_masks_tensor, all_scores_tensor)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=15)

### Classify sequence using `bertForSequenceClassification`
# Load bertForSequenceClassification
model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForSequenceClassification', 'bert-base-chinese', num_labels=num_labels)
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Get trainable parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Calc total steps
t_total = len(train_dataloader) * EPOCH

# Initialize optimizer
optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warnup=0.1, t_total=t_total)

global_step = 0

model.train()
for _ in trange(EPOCH, desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        tokens_ids, segments_ids, masks, scores = batch

        # define a new function to compute loss values for both output_modes
        logits = model(tokens_ids, segments_ids, attention_mask=masks)

        loss = cross_entropy(logits.view(-1, num_labels), scores.view(-1, num_labels))

        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += tokens_ids.size(0)
        nb_tr_steps += 1

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(OUTPUT_DIR)

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
