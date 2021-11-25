import os
import pickle

from utils.Dataloader import Dataloader

##
# Dictinoary class - the dictionary for the RNN models. Holds the language and char vectors for the dataset.
# #
class Dictionary(object):
    def __init__(self):
        self.indicies = {}
        self.tokens = []

    # Checks whether the token is already added or not.
    def new_token(self, token):
        # Adds the token to the list, if not already added
        # The index is set to the length of the token array - 1
        if token not in self.indicies:
            self.tokens.append(token)
            self.indicies[token] = len(self.tokens) - 1
        return self.indicies[token]

    def __len__(self):
        return len(self.tokens)

# Create new dictionary and adds tokens and language
def create_dictionary(x, y):

    char_dictionary = Dictionary()
    pad_token = '<pad>'  # reserve index 0 for padding
    unk_token = '<unk>'  # reserve index 1 for unknown token
    char_dictionary.new_token(pad_token)
    char_dictionary.new_token(unk_token)

    chars = set(''.join(x))
    for char in sorted(chars):
        char_dictionary.new_token(char)
    print("Character vocabulary:", len(char_dictionary), "UTF characters")

    language_dictionary = Dictionary()
    languages = set(y)
    for lang in sorted(languages):
        language_dictionary.new_token(lang)
    print("Language vocabulary:", len(language_dictionary), "languages")

    return char_dictionary, language_dictionary


# Takes the language and char dictionaries created in memory and writes them to a .txt file for use in inference later
def write_dictionary(path, x, y):
    if not os.path.exists(path):
        os.makedirs(path)

    char_dictionary, language_dictionary = create_dictionary(x, y)
    with open(f'{path}char_dic.txt', 'wb') as f1:
        pickle.dump(char_dictionary, f1)
    with open(f'{path}lang_dic.txt', 'wb') as f2:
        pickle.dump(language_dictionary, f2)

    return char_dictionary, language_dictionary

# Loads the two dictionary for when a using a model for inference
def load_dictionary(path):
    with open(f'{path}char_dic.txt', 'rb') as f1:
        char_dic = pickle.load(f1)
    with open(f'{path}lang_dic.txt', 'rb') as f2:
        lang_dic = pickle.load(f2)

    return char_dic, lang_dic
