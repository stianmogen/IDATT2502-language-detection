import random
import numpy as np
import pandas as pd
import torch
import torch.nn
import time
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import os

from rnn.rnn_model import CharRNNClassifier

seed = 1111
random.seed(seed)
np.random.RandomState(seed)

device = ""
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0")
else:
    torch.manual_seed(seed)
    device = torch.device("cpu")

print(device)


def read_file(dir, file, encoding="utf8"):
    with open(os.path.join(dir, file), encoding=encoding) as f:
        data = f.read()
        return data


INPUT_DIR = "../input/"

x_train = read_file(INPUT_DIR, "x_train.txt").split('\n')
y_train = read_file(INPUT_DIR, "y_train.txt").split('\n')
x_train.pop(-1)
y_train.pop(-1)

x_test = read_file(INPUT_DIR, "x_test.txt").split('\n')
y_test = read_file(INPUT_DIR, "y_test.txt").split('\n')
x_test.pop(-1)
y_test.pop(-1)

dataset_x = x_train + x_test
dataset_y = y_train + y_test

dataset_x = pd.DataFrame(dataset_x, columns=['sentence'])
dataset_y = pd.DataFrame(dataset_y, columns=['language'])

print(dataset_x.shape)
print(dataset_y.shape)

print('Example:')
print('LANG =', dataset_y['language'].iloc[0])
print('TEXT =', dataset_x['sentence'].iloc[0])

x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=42)

x_train_sentence = x_train['sentence']
y_train_language = y_train['language']


class Dictionary(object):
    def __init__(self):
        self.indicies = {}
        self.tokens = []

    def new_token(self, token):
        # Adds the token to the list, if not already added
        # The index is set to the length of the token array - 1
        if token not in self.indicies:
            self.tokens.append(token)
            self.indicies[token] = len(self.tokens) - 1
        return self.indicies[token]

    def __len__(self):
        return len(self.tokens)


char_dictionary = Dictionary()
pad_token = '<pad>'  # reserve index 0 for padding
unk_token = '<unk>'  # reserve index 1 for unknown token
pad_index = char_dictionary.new_token(pad_token)
unk_index = char_dictionary.new_token(unk_token)

chars = set(''.join(x_train_sentence))
for char in sorted(chars):
    char_dictionary.new_token(char)
print("Vocabulary:", len(char_dictionary), "UTF characters")

language_dictionary = Dictionary()
# use python set to obtain the list of languages without repetitions
languages = set(y_train_language)
for lang in sorted(languages):
    language_dictionary.new_token(lang)
print("Labels:", len(language_dictionary), "languages")

x_train_idx = [np.array([char_dictionary.indicies[c] for c in line]) for line in x_train_sentence]
y_train_idx = np.array([language_dictionary.indicies[lang] for lang in y_train_language])

x_train_idx, x_val_idx, y_train_idx, y_val_idx = train_test_split(x_train_idx, y_train_idx, test_size=0.125,
                                                                  random_state=42)

train_data = [(x, y) for x, y in zip(x_train_idx, y_train_idx)]
val_data = [(x, y) for x, y in zip(x_val_idx, y_val_idx)]

print(len(train_data), "training samples")
print(len(val_data), "validation samples")


def batch_generator(data, batch_size, token_size):
    """Yield elements from data in chunks with a maximum of batch_size sequences and token_size tokens."""
    minibatch, sequences_so_far, tokens_so_far = [], 0, 0
    for ex in data:
        seq_len = len(ex[0])
        if seq_len > token_size:
            ex = (ex[0][:token_size], ex[1])
            seq_len = token_size
        minibatch.append(ex)
        sequences_so_far += 1
        tokens_so_far += seq_len
        if sequences_so_far == batch_size or tokens_so_far == token_size:
            yield minibatch
            minibatch, sequences_so_far, tokens_so_far = [], 0, 0
        elif sequences_so_far > batch_size or tokens_so_far > token_size:
            yield minibatch[:-1]
            minibatch, sequences_so_far, tokens_so_far = minibatch[-1:], 1, len(minibatch[-1][0])
    if minibatch:
        yield minibatch


def pool_generator(data, batch_size, token_size, shuffle=False):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*token_size, sorts examples within
    each chunk, then batch these examples and shuffle the batches.
    """
    for p in batch_generator(data, batch_size * 100, token_size * 100):
        p_batch = batch_generator(sorted(p, key=lambda t: len(t[0]), reverse=True), batch_size, token_size)
        p_list = list(p_batch)
        if shuffle:
            for b in random.sample(p_list, len(p_list)):
                yield b
        else:
            for b in p_list:
                yield b


criterion = torch.nn.CrossEntropyLoss(reduction='sum')


def train(model, optimizer, data, batch_size, token_size, max_norm=1, log=False):
    model.train()
    total_loss = 0
    ncorrect = 0
    nsentences = 0
    ntokens = 0
    niterations = 0
    for batch in pool_generator(data, batch_size, token_size, shuffle=True):
        # Get input and target sequences from batch
        X = [torch.from_numpy(d[0]) for d in batch]
        X_lengths = [x.numel() for x in X]
        ntokens += sum(X_lengths)
        X_lengths = torch.tensor(X_lengths, dtype=torch.long)
        y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
        # Pad the input sequences to create a matrix
        X = torch.nn.utils.rnn.pad_sequence(X).to(device)
        model.zero_grad()
        output = model(X, X_lengths)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm)  # Gradient clipping https://www.kaggle.com/c/wili4/discussion/231378
        optimizer.step()
        # Training statistics
        total_loss += loss.item()
        ncorrect += (torch.max(output, 1)[1] == y).sum().item()
        nsentences += y.numel()
        niterations += 1

    total_loss = total_loss / nsentences
    accuracy = 100 * ncorrect / nsentences
    if log:
        print(f'Train: wpb={ntokens // niterations}, bsz={nsentences // niterations}, num_updates={niterations}')
    return accuracy


def validate(model, data, batch_size, token_size):
    model.eval()
    # calculate accuracy on validation set
    ncorrect = 0
    nsentences = 0
    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input and target sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long)
            y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)
            answer = model(X, X_lengths)
            ncorrect += (torch.max(answer, 1)[1] == y).sum().item()
            nsentences += y.numel()
        dev_acc = 100 * ncorrect / nsentences
    return dev_acc


hidden_size = 150
embedding_size = 64
bidirectional = False
ntokens = len(char_dictionary)
nlabels = len(language_dictionary)

model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, pad_idx=pad_index,
                          bidirectional=bidirectional).to(device)
optimizer = torch.optim.Adam(model.parameters())

batch_size, token_size = 256, 200000
epochs = 4
train_accuracy = []
valid_accuracy = []

print(f'Training cross-validation model for {epochs} epochs')
t0 = time.time()
for epoch in range(1, epochs + 1):
    acc = train(model, optimizer, train_data, batch_size, token_size, log=epoch == 1)
    train_accuracy.append(acc)
    print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}% ({time.time() - t0:.0f}s)')
    acc = validate(model, val_data, batch_size, token_size)
    valid_accuracy.append(acc)
    print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%')

print(model)
for name, param in model.named_parameters():
    print(f'{name:20} {param.numel()} {list(param.shape)}')
print(f'TOTAL                {sum(p.numel() for p in model.parameters())}')

plt.plot(range(1, len(train_accuracy) + 1), train_accuracy)
plt.plot(range(1, len(valid_accuracy) + 1), valid_accuracy)
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.show()
