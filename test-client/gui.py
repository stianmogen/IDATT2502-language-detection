import pickle
import re

import PySimpleGUI as sg
import numpy as np
import torch
from pandas import read_csv

from rnn.dictionary import load_dictionary
from rnn.rnn_model import CharRNNClassifier

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

sg.theme('HotDogStand')

mnb_out_root = "../mnb/out/"
rnn_out_root = "../rnn/out/"

char_dictionary, lang_dictionary = load_dictionary("../rnn/out/")
labels = read_csv("../input/labels.csv", sep=';')


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
    preds = []
    for i in range(len(top_preds)):
        lang = lang_dictionary.tokens[top_preds[i]]
        lang = labels[labels.Label == lang].English.item()
        preds.append(lang)
    return preds


def mnb_predict(text, cv, le, model):
    text = remove_special(text)
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    answer = torch.tensor(model.predict_proba(x))  # predicting the language
    top_preds = torch.topk(answer, 3, 1)[1][0]
    preds = []
    for i in range(len(top_preds)):
        lang = le.inverse_transform([top_preds[i]])
        lang = labels[labels.Label == lang[0]].English.item()
        preds.append(lang)
    return preds


def remove_special(sentence):
    return re.sub(r'[\\/:*«`\'?¿";!<>,.|()-_)}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', '', sentence.lower().strip())


def get_model(name):
    if name == models[0]:
        return rnn_model("lstm", 128, True)
    elif name == models[1]:
        return rnn_model("lstm", 256, True)
    elif name == models[2]:
        return rnn_model("lstm", 512, True)
    elif name == models[3]:
        return rnn_model("gru", 128, True)
    elif name == models[4]:
        return rnn_model("gru", 256, True)
    elif name == models[5]:
        return rnn_model("gru", 512, True)
    elif name == models[6]:
        return rnn_model("lstm", 128, False)
    elif name == models[7]:
        return rnn_model("lstm", 256, False)
    elif name == models[8]:
        return rnn_model("lstm", 512, False)
    elif name == models[9]:
        return rnn_model("gru", 128, False)
    elif name == models[10]:
        return rnn_model("gru", 256, False)
    elif name == models[11]:
        return rnn_model("gru", 512, False)
    elif name == models[12]:
        return mnb_model("char", "11")
    elif name == models[13]:
        return mnb_model("char", "21")
    elif name == models[14]:
        return mnb_model("char", "31")
    elif name == models[15]:
        return mnb_model("char", "22")
    elif name == models[16]:
        return mnb_model("char", "32")
    elif name == models[17]:
        return mnb_model("char", "33")
    elif name == models[18]:
        return mnb_model("word", "11")


models = ['Bi_Lstm_128', 'Bi_Lstm_256', 'Bi_Lstm_512', 'Bi_Gru_128', 'Bi_Gru_256', 'Bi_Gru_512', 'Uni_Lstm_128', 'Uni_Lstm_256', 'Uni_Lstm_512', 'Uni_Gru_128', 'Uni_Gru_256', 'Uni_Gru_512', 'Char11', 'Char21', 'Char31', 'Char22', 'Char32', 'Char33', 'Word11']
current_model = 'Bi_Lstm_512'

layout = [[sg.Text('Choose model')],
          [sg.Listbox(values=models, default_values='Bi_Lstm_512', select_mode='single', key='-MODEL-', size=(30, 6))],
          [sg.Input(key='-IN-')], [sg.Button('Show'), sg.Button('Exit')],
          [sg.Text('Prediction:')],
          [sg.Text(size=(100,3), key='-OUTPUT-')]]

window = sg.Window('Pattern 2B', layout, size=(500, 500))

model = get_model(current_model)

while True:  # Event Loop
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    elif event == 'Show':
        print(values['-IN-'])
        if current_model != values['-MODEL-'][0]:
            model = get_model(values['-MODEL-'][0])
        if models.index(current_model) < 12:
            predictions = rnn_predict(values['-IN-'], model)
        else:
            predictions = mnb_predict(values['-IN-'], model.cv, model.le, model)

        print(values['-MODEL-'][0])
        p = ""
        for i in range(3):
            p += f'{i+1}. {predictions[i]}\n'
        window['-OUTPUT-'].update(p)
        print(p)

window.close()




