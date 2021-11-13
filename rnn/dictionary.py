import pickle

from utils.Dataloader import Dataloader


class Dictionary(object):
    def __init__(self):
        self.indicies = {}
        self.tokens = []

    def new_token(self, token):
        # Adds the token to the list, if not already added
        # The index is set to the length of the token array - 1
        if token not in self.indicies:
            self.tokens.append(token)
            self.indicies[token] = len(self.tokens) - 1
        return self.indicies[token]

    def __len__(self):
        return len(self.tokens)


def create_dictionary():
    INPUT_DIR = "../input/dataset"
    dataloader = Dataloader(INPUT_DIR)
    x_train, y_train, _, _, _, _ = dataloader.get_dataframes()

    char_dictionary = Dictionary()
    pad_token = '<pad>'  # reserve index 0 for padding
    unk_token = '<unk>'  # reserve index 1 for unknown token
    pad_index = char_dictionary.new_token(pad_token)
    unk_index = char_dictionary.new_token(unk_token)

    chars = set(''.join(x_train["sentence"]))
    for char in sorted(chars):
        char_dictionary.new_token(char)
    print("Character vocabulary:", len(char_dictionary), "UTF characters")

    language_dictionary = Dictionary()
    languages = set(y_train["language"])
    for lang in sorted(languages):
        language_dictionary.new_token(lang)
    print("Language vocabulary:", len(language_dictionary), "languages")

    return char_dictionary, language_dictionary


def write_dictionary():
    root_out_path = "out/"
    char_dictionary, language_dictionary = create_dictionary()
    with open(f'{root_out_path}char_dic.txt', 'wb') as f1:
        pickle.dump(char_dictionary, f1)
    with open(f'{root_out_path}lang_dic.txt', 'wb') as f2:
        pickle.dump(language_dictionary, f2)


def load_dictionary():
    root_out_path = "out/"
    with open(f'{root_out_path}char_dic.txt', 'rb') as f1:
        char_dic = pickle.load(f1)
    with open(f'{root_out_path}lang_dic.txt', 'rb') as f2:
        lang_dic = pickle.load(f2)

    return char_dic, lang_dic