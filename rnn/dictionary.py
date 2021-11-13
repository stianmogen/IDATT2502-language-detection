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
    x_train, y_train, x_val, y_val, x_test, y_test = dataloader.get_dataframes()

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

create_dictionary()