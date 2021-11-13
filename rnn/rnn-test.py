import torch

from rnn.rnn_model import CharRNNClassifier

PATH = "out/lstm/512/model15.pth"
char_dictionary =
ntokens = len(char_dictionary)
input_size = ntokens
embedding_size = 64
hidden_size = 512
output_size = 235
model = "lstm"
num_layers = 1

model = CharRNNClassifier(input_size=input_size, embedding_size=embedding_size, hidden_size=hidden_size, output_size=output_size, model=model, num_layers=num_layers)
model.load_state_dict(torch.load(PATH))
model.eval()