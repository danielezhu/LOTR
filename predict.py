import train
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import os
import pickle
from argparse import Namespace

FILE = 'checkpoint_pt/model-6000.pth'
model = train.Model(20118, train.flags.seq_size, train.flags.embedding_size, train.flags.lstm_hidden_size)
model.load_state_dict(torch.load(FILE))


def predict(device, model, initial_words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    model.eval()
    hidden_state, cell_state = model.zero_state(1)
    hidden_state = hidden_state.to(device)
    cell_state = cell_state.to(device)

    # Run all of the initial words through the model
    for w in initial_words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (hidden_state, cell_state) = model(ix, (hidden_state, cell_state))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])
    initial_words.append(int_to_vocab[choice])

    sample_len = 100
    # Repeatedly sample
    for _ in range(sample_len):
        ix = torch.tensor([[choice]]).to(device)
        output, (hidden_state, cell_state) = model(ix, (hidden_state, cell_state))
        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        initial_words.append(int_to_vocab[choice])

    print(' '.join(initial_words).encode('utf-8'))

def main():
    user_input = input("Input the start to the sentence:")
    initial_words = user_input.split()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_to_int = pickle.load(open("vocab_to_int.p", "rb"))
    int_to_vocab = pickle.load(open("int_to_vocab.p", "rb"))
    predict(device, model, initial_words, 20118, vocab_to_int, int_to_vocab)


if __name__ == '__main__':
    main()
