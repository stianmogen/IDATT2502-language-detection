import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from sklearn.metrics import confusion_matrix

from rnn.dictionary import load_dictionary
from rnn.rnn_model import CharRNNClassifier
from utils.Dataloader import Dataloader
from utils.confusion import max_deviation
from utils.model_validator import validate

device = ""
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(device)

char_dictionary, lang_dictionary = load_dictionary("out/")

model = "lstm"
hidden_size = 512
bidirectional = True

direction = "unidirectional"
if bidirectional:
    direction = "bidirectional"

PATH = "out/" + direction + "/" + model + "/" + str(hidden_size) + "/model.pth"

ntokens = len(char_dictionary)
input_size = ntokens
embedding_size = 64
output_size = 235
num_layers = 1
batch_size = 64

model = CharRNNClassifier(input_size=input_size, embedding_size=embedding_size, hidden_size=hidden_size,
                          output_size=output_size, model=model, num_layers=num_layers, bidirectional=bidirectional)
model.load_state_dict(torch.load(PATH))
model = model.to(device)
model.eval()

INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
_, _, _, _, x_test, y_test = dataloader.get_dataframes()

x_test_idx = [np.array([char_dictionary.indicies[c] for c in line if c in char_dictionary.indicies]) for line in
              x_test["sentence"]]
y_test_idx = np.array([lang_dictionary.indicies[lang] for lang in y_test["language"]])

criterion = torch.nn.CrossEntropyLoss(reduction='sum')

test_data = [(x, y) for x, y in zip(x_test_idx, y_test_idx)]


def print_confusion_matrix():
    acc, loss, y_pred, y_actual = validate(model, criterion, test_data, batch_size, 1200, device)

    cm = confusion_matrix(y_actual, y_pred)

    """
    Print complete confusion matrix
    plt.figure(figsize=(100, 75))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(lang_dictionary.tokens)
    ax.yaxis.set_ticklabels(lang_dictionary.tokens)
    plt.savefig("confusionmatrix_complete.png")
    plt.show()
    """

    max_deviations = max_deviation(cm, 10)

    y_p, y_a = [], []
    for i in range(len(y_pred)):
        if (y_pred[i] in max_deviations) and (y_actual[i] in max_deviations):
            y_p.append(y_pred[i])
            y_a.append(y_actual[i])

    cm = confusion_matrix(y_a, y_p)

    plt.figure(figsize=(12, 9))

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    labels = []

    for m in max_deviations:
        labels.append(lang_dictionary.tokens[m])

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.savefig("confusionmatrix_small.png")
    plt.show()
    print(acc, loss)


def print_acc_by_seq_len():
    list_y = []
    list_x = []

    for i in range(1, 41):
        token_size = i * 10
        acc, _, _, _ = validate(model, criterion, test_data, batch_size, token_size, device)
        list_y.append(acc)
        list_x.append(token_size)

    plt.plot(list_x, list_y)
    plt.xlabel('Paragraph max length')
    plt.ylabel('Accuracy')
    plt.savefig('len_test_acc_rnn.png')
    plt.show()


print_confusion_matrix()
print_acc_by_seq_len()
