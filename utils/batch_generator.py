import random


def batch_generator(data, batch_size, token_size):
    "Yield elements from data in chunks with a maximum of batch_size sequences"
    minibatch, sequences_count = [], 0
    for ex in data:
        seq_len = len(ex[0])
        if seq_len > token_size:
            ex = (ex[0][:token_size], ex[1])
        minibatch.append(ex)
        sequences_count += 1
        if sequences_count == batch_size:
            yield minibatch
            minibatch, sequences_count = [], 0
        elif sequences_count > batch_size:
            yield minibatch[:-1]
            minibatch, sequences_count = minibatch[-1:], 1
    if minibatch:
        yield minibatch


def pool_generator(data, batch_size, token_size, shuffle=False):
    "Divides into buckets of 100 * batchsize -> sorts within each bucket -> sends batches of size batchsize"
    for p in batch_generator(data, batch_size * 100, token_size * 100):
        p_batch = batch_generator(sorted(p, key=lambda t: len(t[0]), reverse=True), batch_size, token_size)
        p_list = list(p_batch)
        if shuffle:
            for b in random.sample(p_list, len(p_list)):
                yield b
        else:
            for b in p_list:
                yield b
