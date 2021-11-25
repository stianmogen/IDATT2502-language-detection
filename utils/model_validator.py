import torch

from utils.batch_generator import pool_generator

"""
Checks the accuracy of a model without optimizing parameters
"""
def validate(model, criterion, data, batch_size, token_size, device):
    model.eval()

    total_loss = 0
    ncorrect = 0
    nsentences = 0
    y_pred = []
    y_actual = []

    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input and target sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long)
            y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)

            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)

            answer = model(X, X_lengths)

            for max in torch.max(answer, 1)[1]:
                y_pred.append(max.item())
            for value in y:
                y_actual.append(value.item())

            loss = criterion(answer, y)

            total_loss += loss.item()
            ncorrect += (torch.max(answer, 1)[1] == y).sum().item()
            nsentences += y.numel()

        total_loss = total_loss / nsentences
        dev_acc = 100 * ncorrect / nsentences
    return dev_acc, total_loss, y_pred, y_actual