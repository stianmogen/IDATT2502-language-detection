import pickle
import re
from operator import index

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from rnn.dictionary import load_dictionary
from rnn.rnn_model import CharRNNClassifier
from utils.Dataloader import Dataloader
from rnn.dictionary import Dictionary

INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
_, _, _, _, x_test, y_test = dataloader.get_dataframes()


def mnb_model(model_path, option_path, x_test, y_test):
    analyzer = "char"
    vectorizer_path = model_path + option_path + "vectorizer"
    label_encoder_path = "../mnb/" + "labelencoder"
    model_path = model_path + option_path + "model.sav"


    with open(vectorizer_path, 'rb') as v:
        cv = pickle.load(v)

    v.close()

    with open(label_encoder_path, 'rb') as l:
        le = pickle.load(l)

    l.close()

    with open(model_path, 'rb') as m:
        model = pickle.load(m)

    m.close()

    x_test = x_test['sentence'].apply(remove_special)
    y_test = y_test['language']

    y_test = le.transform(y_test)
    x_test = cv.transform(x_test)

    y_pred = model.predict(x_test)

    ac = accuracy_score(y_test, y_pred) * 100

    print(f"Accuracy is: {ac:.1f}%")

    return model, cv, le

def rnn_model(model_path, option_path, x_test, y_test):

    device = ""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    char_dictionary, lang_dictionary = load_dictionary("../rnn/out/")

    ntokens = len(char_dictionary)
    input_size = ntokens
    embedding_size = 64
    hidden_size = 512
    output_size = 235
    model = "lstm"
    num_layers = 1
    batch_size, token_size = 64, 1200

    model = CharRNNClassifier(input_size=input_size, embedding_size=embedding_size, hidden_size=hidden_size,
                              output_size=output_size, model=model, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path + option_path + "model.pth"))
    model = model.to(device)
    model.eval()

    return model


def predict(text, cv, le, model):
    text = remove_special(text)
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    lang = le.inverse_transform(lang)  # finding the language corresponding the the predicted value
    print("The langauge is in", lang[0])


def remove_special(sentence):
    return re.sub(r'[\\/:*«`\'?¿";!<>,.|()-_)}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', '', sentence.lower().strip())


model_types = ["../rnn/out/gru/", "../rnn/out/lstm/", "../mnb/out/"]
hidden_options = ["128/", "256/", "512/"]
mnb_options = ["11/", "21/", "22/", "31/", "32/", "33/"]

model_choice = ""
option_choice = ""
sentence = ""

while model_choice != "q" and model_choice != "q" and sentence != "q":
    sentence = ""
    model_choice = input("Which model: \nGRU: 0\nLSTM: 1\nMNB: 2\n")
    model_path = model_types[int(model_choice)]
    if int(model_choice) == 2:
        option_choice = input("Which n-gram option: \n11: 0\n21: 1\n22: 2\n31: 3\n32: 4\n33: 5\n")
        option_path = mnb_options[int(option_choice)]
        model, cv, le = mnb_model(model_path, option_path, x_test, y_test)
        while sentence != "q" and sentence != "e":
            sentence = input("Enter your sentence, e to exit, q to quit: ")
            predict(sentence, cv, le, model)
    else:
        option_choice = input("Which Hidden Size option for RNN: \n128: 0\n256: 1\n512: 2\n")
        option_path = hidden_options[int(option_choice)]
        model = rnn_model(model_path, option_path, x_test, y_test)
        while sentence != "q" and sentence != "e":
            sentence = input("Enter your sentence, e to exit, q to quit: ")