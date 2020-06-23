import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import os
from argparse import Namespace
flags = Namespace(
train_file='data/book1_cleaned.txt',
seq_size=32,
batch_size=16,
embedding_size=64,
lstm_hidden_size=64,
gradients_norm=5,
initial_words=['I', 'am'],
predict_top_k=5,
checkpoint_path='checkpoint',
)

def get_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, encoding='utf8') as f:
        text = f.read()

    text = text.split()
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k:w for k, w in enumerate(sorted_vocab)}
    # >>> print(list(int_to_vocab.items())[:5])
    # [(0, 'the'), (1, 'and'), (2, 'of'), (3, 'to'), (4, 'a')]
    vocab_to_int = {w:k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (batch_size * seq_size))
    # truncate int_text so that its length is a multiple of batch_size * seq_size
    in_text = int_text[:num_batches * batch_size * seq_size]

    # output is just input shifted to the left by one step
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]

    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text

def get_batch(in_text, out_text, batch_size, seq_size):
    """
    in_text and out_text are batch_size * (num_batches * seq_size) "matrices"
    of numbers (which represent words). Although sentences will of course
    not all be seq_size in length, we use seq_size to standardize the length
    of our "sentences" (which are basically sentence fragments).
    Thus, the values that are yielded by this method are a single batch
    of "sentences" (Each seq_size-length slice per row in the horizontal dimension
    of the in_text and out_text "matrices" represents a "sentence".
    We are getting every single one of the batch_size rows.)
    """
    num_batches = np.prod(in_text.shape) // (batch_size * seq_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


class Model(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_hidden_size):
        super(Model, self).__init__()
        self.seq_size = seq_size
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_hidden_size, batch_first=True)
        self.dense = nn.Linear(lstm_hidden_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)
        return logits, state

    def zero_state(self, batch_size):
        """
        Return two sets of torch.zeros since we need both the hidden state
        and cell state.
        """
        return (torch.zeros(1, batch_size, self.lstm_hidden_size),
                torch.zeros(1, batch_size, self.lstm_hidden_size))


def get_loss_and_train_op(model, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer


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

    print(' '.join(words).encode('utf-8'))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(flags.train_file, flags.batch_size, flags.seq_size)

    model = Model(n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_hidden_size)
    model = model.to(device)
    criterion, optimizer, = get_loss_and_train_op(model, 0.01)

    iteration, num_epochs = 0, 50
    for epoch in range(num_epochs):
        batch = get_batch(in_text, out_text, flags.batch_size, flags.seq_size)
        hidden_state, cell_state = model.zero_state(flags.batch_size)

        # Transfer data to GPU
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
        for x, y in batch:
            iteration += 1
            # Tell model that we are in training mode; does not actually execute any training.
            model.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits, (hidden_state, cell_state) = model(x, (hidden_state, cell_state))
            loss = criterion(logits.transpose(1, 2), y) # transpose dimensions 1 and 2; dim 0 is batch dimension

            # Remove hidden_state and cell_state from the graph
            # in order to do truncated backprop through time.
            hidden_state = hidden_state.detach()
            cell_state = cell_state.detach()

            loss_value = loss.item()

            # Backward pass, gradient clipping, and update parameters
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), flags.gradients_norm)
            optimizer.step()

            # Print info
            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(epoch, num_epochs),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

            if iteration % 1000 == 0:
                predict(device, model, flags.initial_words, n_vocab,
                        vocab_to_int, int_to_vocab)
                torch.save(model.state_dict(), 'checkpoint_pt/model-{}.pth'.format(iteration))


if __name__ == '__main__':
    main()
