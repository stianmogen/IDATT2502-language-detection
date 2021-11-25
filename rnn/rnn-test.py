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

INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
_, _, _, _, x_test, y_test = dataloader.get_dataframes()

x_test_idx = [np.array([char_dictionary.indicies[c] for c in line if c in char_dictionary.indicies]) for line in
              x_test["sentence"]]
y_test_idx = np.array([lang_dictionary.indicies[lang] for lang in y_test["language"]])

criterion = torch.nn.CrossEntropyLoss(reduction='sum')

test_data = [(x, y) for x, y in zip(x_test_idx, y_test_idx)]


def eval_model(model, criterion, test_data, device, batch_size=64, token_size=1200):
    return validate(model, criterion, test_data, batch_size, token_size, device)


def eval_nynorsk_bokmaal(model, criterion, test_data, device, batch_size=64, token_size=1200):
    norsk_index = [lang_dictionary.indicies["nno"], lang_dictionary.indicies["nob"]]
    norsk = list(filter(lambda s: s[1] in norsk_index, test_data))
    acc, loss, y_pred, y_actual = validate(model, criterion, norsk, batch_size, token_size, device)

    labels = set()
    for y in y_pred:
        labels.add(lang_dictionary.tokens[y])
    for y in y_actual:
        labels.add(lang_dictionary.tokens[y])

    labels = sorted(labels)
    print(labels)
    print("Norsk vs nynorsk")
    print("acc:", acc)
    print("loss:", loss)
    print_acc_by_seq_len(model, criterion, norsk, device, name="norsk_acc_by_seq_len.png")
    print_confusion_matrix(y_pred, y_actual, labels, name="norsk_nynorsk_confusion.png")


def print_confusion_matrix(y_pred, y_actual, labels, figsize=(12, 9), name="confusionmatrix.png"):

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

    cm = confusion_matrix(y_actual, y_pred)

    plt.figure(figsize=figsize)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.savefig(f"out/{name}")
    plt.show()


def print_acc_by_seq_len(model, criterion, test_data, device, name="len_test_acc_rnn.png", batch_size=64):
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
    plt.savefig(f"out/{name}")
    plt.show()


hidden_sizes = [128, 256, 512]
model_types = ["lstm", "gru"]
bidirectional_types = {"bidirectional": True, "unidirectional": False}

ntokens = len(char_dictionary)
input_size = ntokens
embedding_size = 64
output_size = len(lang_dictionary)
num_layers = 1
batch_size = 64

# Using the model with best performance
PATH = "out/bidirectional/lstm/512/model.pth"
model = CharRNNClassifier(input_size=input_size, embedding_size=embedding_size, hidden_size=512,
                          output_size=output_size, model="lstm", num_layers=num_layers,
                          bidirectional=True)
model.load_state_dict(torch.load(PATH))
model = model.to(device)
model.eval()


# Test predictions on norwegian
eval_nynorsk_bokmaal(model, criterion, test_data, device)


# Print confusion matrix with max deviations
_, _, y_pred, y_actual = eval_model(model, criterion, test_data, device)
cm = confusion_matrix(y_actual, y_pred)
max_deviations = max_deviation(cm, 10) # Getting langs with deviation to actual answer more than 10
y_p, y_a = [], []
for i in range(len(y_pred)):
    if (y_pred[i] in max_deviations) and (y_actual[i] in max_deviations):
        y_p.append(y_pred[i])
        y_a.append(y_actual[i])
labels = []
for m in max_deviations:
    labels.append(lang_dictionary.tokens[m])
print_confusion_matrix(y_p, y_a, labels)

# Print model accuracy based on paragraph max length
print_acc_by_seq_len(model, criterion, test_data, device)

# Test all models
for bidirectional_type in bidirectional_types:
    for model_type in model_types:
        for hidden_size in hidden_sizes:
            PATH = "out/" + bidirectional_type + "/" + model_type + "/" + str(hidden_size) + "/model.pth"
            model = CharRNNClassifier(input_size=input_size, embedding_size=embedding_size, hidden_size=hidden_size,
                                      output_size=output_size, model=model_type, num_layers=num_layers,
                                      bidirectional=bidirectional_types[bidirectional_type])
            model.load_state_dict(torch.load(PATH))
            model = model.to(device)
            model.eval()
            acc, loss, y_pred, y_actual = eval_model(model, criterion, test_data, device)
            print(bidirectional_type, model_type, hidden_size)
            print("Accuracy:", acc)
            print("Loss:", loss)
