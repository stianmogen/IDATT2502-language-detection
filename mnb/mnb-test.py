import pickle
import re

from sklearn.metrics import accuracy_score

from utils.Dataloader import Dataloader

INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
_, _, _, _, x_test, y_test = dataloader.get_dataframes()


path = "out/11/"

analyzer = "char"
vectorizer_path = path + "vectorizer"
label_encoder_path = "out/labelencoder"
model_path = path + "model.sav"


with open(vectorizer_path, 'rb') as v:
    cv = pickle.load(v)

v.close()

with open(label_encoder_path, 'rb') as l:
    le = pickle.load(l)

l.close()

with open(model_path, 'rb') as m:
    model = pickle.load(m)

m.close()


def predict(text, cv, model):
    text = remove_special(text)
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    lang = le.inverse_transform(lang)  # finding the language corresponding the the predicted value
    print("The langauge is in", lang[0])


def remove_special(sentence):
    return re.sub(r'[\\/:*«`\'?¿";!<>,.|()-_)}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', '', sentence.lower().strip())


x_test = x_test['sentence'].apply(remove_special)
y_test = y_test['language']

y_test = le.transform(y_test)
x_test = cv.transform(x_test)

y_pred = model.predict(x_test)

ac = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy is: {ac:.1f}%")

choice = ""

while choice != "exit":
    choice = input("Enter your sentence: ")
    predict(choice, cv, model)
