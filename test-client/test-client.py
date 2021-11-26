import pickle
import re

import torch
import numpy as np
from pandas import read_csv

from rnn.dictionary import load_dictionary
from rnn.rnn_model import CharRNNClassifier

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

mnb_out_root = "../mnb/out/"
rnn_out_root = "../rnn/out/"

char_dictionary, lang_dictionary = load_dictionary("../rnn/out/")
labels = read_csv("../input/labels.csv", sep=';')
print(labels[labels.Label=='nob'].English.item())


def mnb_model(type, ngram):
    path = f"{mnb_out_root}{type}/{ngram}/"
    vectorizer_path = f"{path}vectorizer"
    label_encoder_path = f"{mnb_out_root}labelencoder"
    model_path = f"{path}model.sav"

    with open(vectorizer_path, 'rb') as v:
        cv = pickle.load(v)

    v.close()

    with open(label_encoder_path, 'rb') as l:
        le = pickle.load(l)

    l.close()

    with open(model_path, 'rb') as m:
        model = pickle.load(m)

    m.close()

    return model, cv, le


def rnn_model(type, hidden_size, bidirectional):

    input_size = len(char_dictionary)
    embedding_size = 64
    output_size = len(lang_dictionary)
    num_layers = 1

    if bidirectional:
        direction_folder = "bidirectional"
    else:
        direction_folder = "unidirectional"

    model = CharRNNClassifier(input_size=input_size, embedding_size=embedding_size, hidden_size=hidden_size,
                              output_size=output_size, model=type, num_layers=num_layers, bidirectional=bidirectional)
    model.load_state_dict(torch.load(f"{rnn_out_root}{direction_folder}/{type}/{hidden_size}/model.pth"))
    model = model.to(device)
    model.eval()

    return model


def rnn_predict(sentence, model):
    sentence_idx = [np.array([char_dictionary.indicies[c] for c in sentence if c in char_dictionary.indicies])]

    X = [torch.from_numpy(d) for d in sentence_idx]
    X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long)

    # Pad the input sequences to create a matrix
    X = torch.nn.utils.rnn.pad_sequence(X).to(device)

    answer = model(X, X_lengths)
    top_preds = torch.topk(answer, 3, 1)[1][0]
    print("Predictions:")
    for i in range(len(top_preds)):
        lang = lang_dictionary.tokens[top_preds[i]]
        lang = labels[labels.Label == lang].English.item()
        print(f"{i+1}. {lang}")


def mnb_predict(text, cv, le, model):
    text = remove_special(text)
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    answer = torch.tensor(model.predict_proba(x))  # predicting the language
    top_preds = torch.topk(answer, 3, 1)[1][0]
    print("Predictions:")
    for i in range(len(top_preds)):
        lang = le.inverse_transform([top_preds[i]])
        lang = labels[labels.Label == lang].English.item()
        print(f"{i + 1}. {lang[0]}")


def remove_special(sentence):
    return re.sub(r'[\\/:*«`\'?¿";!<>,.|()-_)}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', '', sentence.lower().strip())


rnn_types = ["lstm", "gru"]
directions = {"bidirectional": True, "unidirectional": False}
hidden_options = [512, 256, 128]

analyzers = {"char": [3], "word": [1]}


print("Welcome to test client")
print("q = quit, m = change model")

finished = False
while not finished:
    model_choice = input("Choose model: \nRNN: 1\nMNB: 2\n")
    if model_choice == "q":
        finished = True

    elif model_choice == "1":
        option_choice = input("Select model options for RNN: \nBi_Lstm_128: 1\nBi_Lstm_256: 2\nBi_Lstm_512: 3\nBi_Gru_128: 4\nBi_Gru_256: 5\nBi_Gru_512: 6\nUni_Lstm_128: 7\nUni_Lstm_256: 8\nUni_Lstm_512: 9\nUni_Gru_128: 10\nUni_Gru_256: 11\nUni_Gru_512: 12\n")
        if model_choice == "q":
            finished = True
        elif option_choice == "1":
            model = rnn_model("lstm", 128, True)
        elif option_choice == "2":
            model = rnn_model("lstm", 256, True)
        elif option_choice == "3":
            model = rnn_model("lstm", 512, True)
        elif option_choice == "4":
            model = rnn_model("gru", 128, True)
        elif option_choice == "5":
            model = rnn_model("gru", 256, True)
        elif option_choice == "6":
            model = rnn_model("gru", 512, True)
        elif option_choice == "7":
            model = rnn_model("lstm", 128, False)
        elif option_choice == "8":
            model = rnn_model("lstm", 256, False)
        elif option_choice == "9":
            model = rnn_model("lstm", 512, False)
        elif option_choice == "10":
            model = rnn_model("gru", 128, False)
        elif option_choice == "11":
            model = rnn_model("gru", 256, False)
        elif option_choice == "12":
            model = rnn_model("gru", 512, False)
        else:
            continue
        sentence = ""
        while sentence != "m":
            sentence = input("Enter a sentence ('q' to quit, 'm' to change model)\n")
            if sentence == "q":
                finished = True
            elif sentence == "m":
                break
            else:
                rnn_predict(sentence, model)

    elif model_choice == "2":
        option_choice = input("Select model options for MNB: \nChar11: 1\nChar21: 2\nChar31: 3\nChar22: 4\nChar32: 5\nChar33: 6\nWord11: 7\n")
        if model_choice == "q":
            finished = True
        elif option_choice == "1":
            model, cv, le = mnb_model("char", "11")
        elif option_choice == "2":
            model, cv, le = mnb_model("char", "21")
        elif option_choice == "3":
            model, cv, le = mnb_model("char", "31")
        elif option_choice == "4":
            model, cv, le = mnb_model("char", "22")
        elif option_choice == "5":
            model, cv, le = mnb_model("char", "32")
        elif option_choice == "6":
            model, cv, le = mnb_model("char", "33")
        elif option_choice == "7":
            model, cv, le = mnb_model("word", "11")
        else:
            continue
        sentence = ""
        while sentence != "m":
            sentence = input("Enter a sentence ('q' to quit, 'm' to change model)\n")
            if sentence == "q":
                finished = True
            elif sentence == "m":
                break
            else:
                mnb_predict(sentence, cv, le, model)
