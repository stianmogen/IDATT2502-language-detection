import random
import numpy as np
import torch
import torch.nn
import time

import matplotlib.pyplot as plt
import os

from rnn.rnn_model import CharRNNClassifier
from utils.Dataloader import Dataloader
from rnn.dictionary import write_dictionary
from utils.batch_generator import pool_generator
from utils.create_dir import create_dir
from utils.model_validator import validate

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

INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
x_train, y_train, x_val, y_val, _, _ = dataloader.get_dataframes()

print("Train:")
print(x_train.shape)
print(y_train.shape)

print("Val:")
print(x_val.shape)
print(y_val.shape)

root_out_path = "out/"
if not os.path.exists(root_out_path):
    os.makedirs(root_out_path)

char_dictionary, language_dictionary = write_dictionary(root_out_path, x_train["sentence"], y_train["language"])

pad_index = 0

x_train_idx = [np.array([char_dictionary.indicies[c] for c in line]) for line in x_train["sentence"]]
y_train_idx = np.array([language_dictionary.indicies[lang] for lang in y_train["language"]])

x_val_idx = [np.array([char_dictionary.indicies[c] for c in line if c in char_dictionary.indicies]) for line in
             x_val["sentence"]]
y_val_idx = np.array([language_dictionary.indicies[lang] for lang in y_val["language"]])

train_data = [(x, y) for x, y in zip(x_train_idx, y_train_idx)]
val_data = [(x, y) for x, y in zip(x_val_idx, y_val_idx)]

criterion = torch.nn.CrossEntropyLoss(reduction='sum')


def train(model, optimizer, data, batch_size, token_size, max_norm=1):
    model.train()

    total_loss = 0
    ncorrect = 0
    nsentences = 0

    for batch in pool_generator(data, batch_size, token_size, shuffle=True):
        # Get input and target sequences from batch
        X = [torch.from_numpy(d[0]) for d in batch]
        X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long)
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

    total_loss = total_loss / nsentences
    accuracy = 100 * ncorrect / nsentences
    return accuracy, total_loss


embedding_size = 64
ntokens = len(char_dictionary)
nlabels = len(language_dictionary)
batch_size = 64
token_size = 1200
epochs = 15

hidden_sizes = [128, 256, 512]
model_types = ["lstm", "gru"]
bidirectional_types = {"bidirectional": True, "unidirectional": False}


def run():
    for bidirectional_type in bidirectional_types:
        bi_type_path = create_dir(root_out_path, bidirectional_type)

        for model_type in model_types:
            model_out_path = create_dir(bi_type_path, model_type)

            for hidden_size in hidden_sizes:
                hidden_out_path = create_dir(model_out_path, str(hidden_size))

                train_accuracy = []
                valid_accuracy = []
                train_loss = []
                valid_loss = []
                valid_max = 0

                model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, model=model_type,
                                          pad_idx=pad_index,
                                          bidirectional=bidirectional_types[bidirectional_type]).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
                print(f'Training cross-validation model for {epochs} epochs')

                t0 = time.time()
                for epoch in range(1, epochs + 1):
                    acc, loss = train(model, optimizer, train_data, batch_size, token_size)

                    train_accuracy.append(acc)
                    train_loss.append(loss)

                    print("Learning rate:", scheduler.get_last_lr())
                    scheduler.step()
                    print(
                        f'| epoch {epoch:03d} | train accuracy={acc:.1f}% | train loss={loss} ({time.time() - t0:.0f}s)')

                    acc, loss, _, _ = validate(model, criterion, val_data, batch_size, token_size, device)
                    if acc > valid_max:
                        valid_max = acc
                        torch.save(model.state_dict(), os.path.join(hidden_out_path, f"model{epoch}.pth"))

                    valid_accuracy.append(acc)
                    valid_loss.append(loss)
                    print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}% | valid loss={loss}')
                    time.sleep(2)  # Cooling down gpu

                print(model)
                for name, param in model.named_parameters():
                    print(f'{name:20} {param.numel()} {list(param.shape)}')
                print(f'TOTAL                {sum(p.numel() for p in model.parameters())}')

                plt.plot(range(1, len(train_accuracy) + 1), train_accuracy)
                plt.plot(range(1, len(valid_accuracy) + 1), valid_accuracy)
                plt.xlabel('epoch')
                plt.ylabel('Accuracy')
                plt.savefig(os.path.join(hidden_out_path, f'acc.png'))
                plt.show()

                plt.plot(range(1, len(train_loss) + 1), train_loss)
                plt.plot(range(1, len(valid_loss) + 1), valid_loss)
                plt.xlabel('epoch')
                plt.ylabel('Loss')
                plt.savefig(os.path.join(hidden_out_path, f'loss.png'))
                plt.show()

                time.sleep(60)  # Cooling down gpu


run()
