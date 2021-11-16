import random
import numpy as np
import torch
import pickle

from rnn.dictionary import Dictionary
from rnn.dictionary import load_dictionary
from rnn.rnn_model import CharRNNClassifier
from utils.Dataloader import Dataloader

device = ""
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(device)

char_dictionary, lang_dictionary = load_dictionary("out/")

PATH = "out/gru/512/model.pth"
ntokens = len(char_dictionary)
input_size = ntokens
embedding_size = 64
hidden_size = 512
output_size = 235
model = "gru"
num_layers = 1
batch_size, token_size = 64, 1200

model = CharRNNClassifier(input_size=input_size, embedding_size=embedding_size, hidden_size=hidden_size, output_size=output_size, model=model, num_layers=num_layers)
model.load_state_dict(torch.load(PATH))
model = model.to(device)
model.eval()

INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
_, _, _, _, x_test, y_test = dataloader.get_dataframes()

x_test_idx = [np.array([char_dictionary.indicies[c] for c in line if c in char_dictionary.indicies]) for line in x_test["sentence"]]
y_test_idx = np.array([lang_dictionary.indicies[lang] for lang in y_test["language"]])


def batch_generator(data, batch_size, token_size):
    "Yield elements from data in chunks with a maximum of batch_size sequences"
    minibatch, sequences_count = [], 0
    for ex in data:
        seq_len = len(ex[0])
        if seq_len > token_size:
            ex = (ex[0][:token_size], ex[1])
        minibatch.append(ex)
        sequences_count += 1
        if sequences_count == batch_size:
            yield minibatch
            minibatch, sequences_count = [], 0
        elif sequences_count > batch_size:
            yield minibatch[:-1]
            minibatch, sequences_count = minibatch[-1:], 1
    if minibatch:
        yield minibatch


def pool_generator(data, batch_size, token_size, shuffle=False):
    "Divides into buckets of 100 * batchsize -> sorts within each bucket -> sends batches of size batchsize"
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

def validate(model, criterion, data, batch_size, token_size):
    model.eval()

    total_loss = 0
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
            loss = criterion(answer, y)

            # Validation statistics
            total_loss += loss.item()
            ncorrect += (torch.max(answer, 1)[1] == y).sum().item()
            nsentences += y.numel()

        total_loss = total_loss / nsentences
        dev_acc = 100 * ncorrect / nsentences
    return dev_acc, total_loss


test_data = [(x, y) for x, y in zip(x_test_idx, y_test_idx)]
acc, loss = validate(model, criterion, test_data, batch_size, token_size)

print(acc, loss)
