import os
import re

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import time

import pandas as pd
import torch

from model import Predictor

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Constants

NUM_CLASSES = 235
INPUT_DIR = "input/"
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 256

with open(os.path.join(INPUT_DIR, "x_train.txt"), encoding="utf8") as f:
    data = f.read()

x_train = data.split('\n')

with open(os.path.join(INPUT_DIR, "x_test.txt"), encoding="utf8") as f:
    data = f.read()

x_test = data.split('\n')

with open(os.path.join(INPUT_DIR, "y_train.txt"), encoding="utf8") as f:
    data = f.read()

y_train = data.split('\n')

with open(os.path.join(INPUT_DIR, "y_test.txt"), encoding="utf8") as f:
    data = f.read()

y_test = data.split('\n')

x_train.pop(-1)
x_test.pop(-1)
y_train.pop(-1)
y_test.pop(-1)


x_train = pd.DataFrame(x_train, columns=['sentence'])
y_train = pd.DataFrame(y_train, columns=['language'])

x_test = pd.DataFrame(x_test, columns=['sentence'])
y_test = pd.DataFrame(y_test, columns=['language'])

print(x_train.shape)
print(x_test.shape)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)


def remove_special(sentence):
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|()-_)(}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', ' ', sentence.lower().strip())


cv = CountVectorizer(ngram_range=(3, 3))

x_train = cv.fit_transform(x_train['sentence'].apply(remove_special))
x_test = cv.transform(x_test['sentence'].apply(remove_special))

print(x_train.shape)
print(x_test.shape)

x_train = torch.from_numpy(x_train.todense()).float()
x_test = torch.from_numpy(x_test.todense()).float()
y_train = torch.from_numpy(np.array(y_train))
y_test = torch.from_numpy(np.array(y_test))

print(x_train.shape, x_test.shape)

model = Predictor(x_train.shape[1], NUM_CLASSES).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, x, y, optimizer, criterion):
    model.train()

    images = x
    labels = y

    optimizer.zero_grad()

    output = model(images)
    loss = criterion(output, labels.argmax(1))

    loss.backward()

    acc = calculate_accuracy(output, labels.argmax(1))

    optimizer.step()

    epoch_loss = loss.item()
    epoch_acc = acc.item()

    return epoch_loss, epoch_acc


def evaluate(model, x, y, criterion):
    model.eval()

    with torch.no_grad():
        images = x
        labels = y

        output = model(images)
        loss = criterion(output, labels.argmax(1))

        acc = calculate_accuracy(output, labels.argmax(1))

        epoch_loss = loss.item()
        epoch_acc = acc.item()

        return epoch_loss, epoch_acc


train_loss_list = [0] * EPOCHS
train_acc_list = [0] * EPOCHS
test_loss_list = [0] * EPOCHS
test_acc_list = [0] * EPOCHS

for epoch in range(EPOCHS):
    print("Epoch:", epoch)

    train_start_time = time.monotonic()
    train_loss, train_acc = train(model, x_train, y_train, optimizer, criterion)
    train_end_time = time.monotonic()

    test_start_time = time.monotonic()
    test_loss, test_acc = evaluate(model, x_test, y_test, criterion)
    test_end_time = time.monotonic()

    train_loss_list[epoch] = train_loss
    train_acc_list[epoch] = train_acc
    test_loss_list[epoch] = test_loss
    test_acc_list[epoch] = test_acc

    print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (
        train_loss, train_acc, train_end_time - train_start_time))
    print("Testing: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (
        test_loss, test_acc, test_end_time - test_start_time))
    print("")
