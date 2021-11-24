import pickle
import re

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from utils.Dataloader import Dataloader

INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
_, _, _, _, x_test, y_test = dataloader.get_dataframes()


def predict(text, cv, model):
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    lang = le.inverse_transform(lang)  # finding the language corresponding the the predicted value
    print("The langauge is in", lang[0])


analyzers = {"char": [3], "word": [1]}

root_path = "out/"
label_encoder_path = root_path + "labelencoder"
with open(label_encoder_path, 'rb') as l:
    le = pickle.load(l)

l.close()

for analyzer in analyzers:
    for max_value in analyzers[analyzer]:
        min_value = max_value
        current_path = f"{root_path}{analyzer}/{max_value}{min_value}/"
        vectorizer_path = current_path + "vectorizer"

        model_path = current_path + "model.sav"

        with open(vectorizer_path, 'rb') as v:
            cv = pickle.load(v)

        v.close()

        with open(model_path, 'rb') as m:
            model = pickle.load(m)

        m.close()
        list_x = []
        list_y = []
        print(f"{analyzer}{max_value}{min_value}")

        for i in range(1, 41):
            step = 10 * i
            print(step)

            def remove_special(sentence):
                if len(sentence) > step:
                    sentence = sentence[0:step]
                return re.sub(r'[\\/:*«`\'?¿";!<>,.|()-_)}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', '',
                              sentence.lower().strip())


            x = x_test['sentence'].apply(remove_special)
            y = y_test['language']

            y = le.transform(y)
            x = cv.transform(x)

            y_pred = model.predict(x)

            ac = accuracy_score(y, y_pred) * 100
            list_x.append(step)
            list_y.append(ac)
        plt.plot(list_x, list_y, label=f'{analyzer}_{max_value}{min_value}')

plt.legend(loc=1)
plt.xlabel('Paragraph max length')
plt.ylabel('Accuracy')
plt.savefig("len_test_acc_rnn.png")
plt.show()

choice = ""

while choice != "exit":
    choice = input("Enter your sentence: ")
    predict(choice, cv, model)
