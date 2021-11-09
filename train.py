import os
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import pandas as pd


# Constants

INPUT_DIR = "input/"

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

print("x_train, x_test shape:")
print(x_train.shape)
print(x_test.shape)


def remove_special(sentence):
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|()-_)(}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', ' ', sentence.lower().strip())


x_train = x_train['sentence'].apply(remove_special)
y_train = y_train['language']

x_test = x_test['sentence'].apply(remove_special)
y_test = y_test['language']


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

cv = CountVectorizer(analyzer='char', ngram_range=(1, 4))

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

print("\nVectorized x_train, x_test: ")
print(x_train.shape)
print(x_test.shape)

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

ac = accuracy_score(y_test, y_pred)

print("Accuracy is :", ac)


def predict(text):
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    lang = le.inverse_transform(lang)  # finding the language corresponding the the predicted value
    print("The langauge is in", lang[0])


predict("my name is Simon")

predict("mi nombre es Simon")

predict("Hei jeg heter Simon")

predict("Her vil vi i dagane framover dele dikt av Olav H. Hauge. Vi vil la Hauges poesi gje våre eigne tankar både pause og perspektiv. Gjer som Hauge, og lat diktet vere ei lauvhytte eller eit snøhus som du kan krype inn i, og vere i, i stille kveldar.")

predict("내 이름은 zisun")

predict("меня зовут цисун")
