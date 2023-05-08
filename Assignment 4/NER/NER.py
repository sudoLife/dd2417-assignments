#!/usr/bin/env python
# coding: utf-8
import argparse
import string
import codecs
import csv
from tqdm import tqdm
from terminaltables import AsciiTable
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from GRU import GRU2

PADDING_WORD = '<pad>'
UNKNOWN_WORD = '<unk>'
CHARS = ['<unk>', '<space>', '’', '—'] + list(string.punctuation) + list(string.ascii_letters) + list(string.digits)


def load_glove_embeddings(embedding_file, padding_idx=0, padding_word=PADDING_WORD, unknown_word=UNKNOWN_WORD):
    """
    The function to load GloVe word embeddings

    Args:
        embedding_file (str): The name of the txt file containing GloVe word embeddings
        padding_idx (int): The index, where to insert padding and unknown words
        padding_word (str): The symbol used as a padding word
        unknown_word (str): The symbol used for unknown words

    Returns:
        tuple: A 4-tuple of (vocabulary size, vector dimensionality, embedding matrix, mapping from words to indices)
    """
    word2index, embeddings, N = {}, [], 0
    with open(embedding_file, encoding='utf8') as f:
        for line in f:
            data = line.split()
            word = data[0]
            vec = [float(x) for x in data[1:]]
            embeddings.append(vec)
            word2index[word] = N
            N += 1
    D = len(embeddings[0])

    if padding_idx is not None and type(padding_idx) is int:
        embeddings.insert(padding_idx, [0]*D)
        embeddings.insert(padding_idx + 1, [-1]*D)
        for word in word2index:
            if word2index[word] >= padding_idx:
                word2index[word] += 2
        word2index[padding_word] = padding_idx
        word2index[unknown_word] = padding_idx + 1

    return N, D, np.array(embeddings, dtype=np.float32), word2index


class NERDataset(Dataset):
    """
    A class loading NER dataset from a CSV file to be used as an input to PyTorch DataLoader.
    """

    def __init__(self, filename):
        reader = csv.reader(codecs.open(filename, encoding='ascii', errors='ignore'), delimiter=',')

        self.sentences = []
        self.labels = []

        sentence, labels = [], []
        for row in reader:
            if row:
                if row[0].strip():
                    if sentence and labels:
                        self.sentences.append(sentence)
                        self.labels.append(labels)
                    sentence = [row[1].strip()]
                    labels = [self.__bio2int(row[3].strip())]
                else:
                    sentence.append(row[1].strip())
                    labels.append(self.__bio2int(row[3].strip()))

    def __bio2int(self, x):
        return 0 if x == 'O' else 1

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class PadSequence:
    """
    A callable used to merge a list of samples to form a padded mini-batch of Tensor
    """

    def __call__(self, batch, pad_data=PADDING_WORD, pad_labels=0):
        batch_data, batch_labels = zip(*batch)
        max_len = max(map(len, batch_labels))
        padded_data = [[b[i] if i < len(b) else pad_data for i in range(max_len)] for b in batch_data]
        padded_labels = [[l[i] if i < len(l) else pad_labels for i in range(max_len)] for l in batch_labels]
        return padded_data, padded_labels


class NERClassifier(nn.Module):
    def __init__(self, word_emb_file, char_emb_size=16, char_hidden_size=25, word_hidden_size=100,
                 padding_word=PADDING_WORD, unknown_word=UNKNOWN_WORD, char_map=CHARS,
                 char_bidirectional=True, word_bidirectional=True):
        """
        Constructs a new instance.

        Args:
            word_emb_file (str): The filename of the file with pre-trained word embeddings
            char_emb_size (int): The character embedding size
            char_hidden_size (int): The character-level BiRNN hidden size
            word_hidden_size (int): The word-level BiRNN hidden size
            padding_word (str): A token used to pad the batch to equal-sized tensor
            unknown_word (str): A token used for the out-of-vocabulary words
            char_map (list): A list of characters to be considered
        """
        super(NERClassifier, self).__init__()
        self.padding_word = padding_word
        self.unknown_word = unknown_word
        self.char_emb_size = char_emb_size
        self.char_hidden_size = char_hidden_size
        self.word_hidden_size = word_hidden_size
        self.char_bidirectional = char_bidirectional
        self.word_bidirectional = word_bidirectional

        vocabulary_size, self.word_emb_size, embeddings, self.w2i = load_glove_embeddings(
            word_emb_file, padding_word=self.padding_word, unknown_word=self.unknown_word
        )

        self.word_emb = nn.Embedding(vocabulary_size, self.word_emb_size)
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=False)

        if self.char_emb_size > 0:
            self.c2i = {c: i for i, c in enumerate(char_map)}
            self.char_emb = nn.Embedding(len(char_map), char_emb_size, padding_idx=0)
            self.char_birnn = GRU2(self.char_emb_size, self.char_hidden_size, bidirectional=char_bidirectional)
        else:
            self.char_hidden_size = 0

        multiplier = 2 if self.char_bidirectional else 1
        self.word_birnn = GRU2(
            self.word_emb_size + multiplier * self.char_hidden_size,  # input size
            self.word_hidden_size,                          # hidden size
            bidirectional=word_bidirectional
        )

        # Binary classification - 0 if not part of the name, 1 if a name
        multiplier = 2 if self.word_bidirectional else 1
        self.final_pred = nn.Linear(multiplier * self.word_hidden_size, 2)

    def forward(self, x):
        """
        Performs a forward pass of a NER classifier
        Takes as input a 2D list `x` of dimensionality (B, T),
        where B is the batch size;
              T is the max sentence length in the batch (the sentences with a smaller length are already padded with a special token <PAD>)

        Returns logits, i.e. the output of the last linear layer before applying softmax.

        :param      x:    A batch of sentences
        :type       x:    list of strings
        """
        #
        # YOUR CODE HERE
        #
        # find the max word length
        # TODO: is this really necessary?
        batch_size = len(x)
        max_sentence_length = len(x[0])

        max_word_length = 0
        for sentence in x:
            for word in sentence:
                max_word_length = max(max_word_length, len(word))

        word_ids = np.zeros((batch_size, max_sentence_length), dtype=int)
        char_ids = np.zeros((batch_size, max_sentence_length, max_word_length), dtype=int)

        for sentence_ind, sentence in enumerate(x):
            for word_ind, word in enumerate(sentence):
                if word.lower() in self.w2i:
                    word_ids[sentence_ind, word_ind] = self.w2i[word.lower()]
                else:
                    # NOTE: could skip this because it's a 0 anyway but for the sake of clean code I'll keep it
                    word_ids[sentence_ind, word_ind] = self.w2i[UNKNOWN_WORD]
                char_ids[sentence_ind, word_ind] = [self.c2i[char]
                                                    for char in word] + [self.c2i[UNKNOWN_WORD]] * (max_word_length - len(word))
        char_ids_tensor = torch.LongTensor(char_ids)
        char_tensor = self.char_emb(char_ids_tensor).view(-1, max_word_length, self.char_emb_size)
        # TODO: fix if unidirectional
        _, h_fw, h_bw = self.char_birnn(char_tensor)

        word_ids_tensor = torch.LongTensor(word_ids)
        word_tensor = self.word_emb(word_ids_tensor)

        input_tensor = torch.concat(
            (word_tensor, h_fw.view(batch_size, max_sentence_length, -1),
             h_bw.view(batch_size, max_sentence_length, -1)),
            dim=-1)

        outputs = self.word_birnn(input_tensor)

        return self.final_pred(outputs[0])


#
# MAIN SECTION
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-tr', '--train', default='data/ner_training.csv',
                        help='A comma-separated training file')
    parser.add_argument('-t', '--test', default='data/ner_test.csv',
                        help='A comma-separated test file')
    parser.add_argument('-wv', '--word-vectors', default='glove.6B.50d.txt',
                        help='A txt file with word vectors')
    parser.add_argument('-c', '--char-emb-size', default=16, type=int,
                        help='A size of char embeddings, put 0 to switch off char embeddings')
    parser.add_argument('-cud', '--char-unidirectional', action='store_true')
    parser.add_argument('-wud', '--word-unidirectional', action='store_true')
    parser.add_argument('-lr', '--learning-rate', default=0.002, help='A learning rate')
    parser.add_argument('-e', '--epochs', default=5, type=int, help='Number of epochs')
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    training_data = NERDataset(args.train)
    training_loader = DataLoader(training_data, batch_size=128, collate_fn=PadSequence())

    ner = NERClassifier(
        args.word_vectors,
        char_emb_size=args.char_emb_size,
        char_bidirectional=not args.char_unidirectional,
        word_bidirectional=not args.word_unidirectional
    )

    optimizer = optim.Adam(ner.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(args.epochs):
        ner.train()
        for x, y in tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
            optimizer.zero_grad()
            logits = ner(x)
            logits_shape = logits.shape

            loss = criterion(logits.reshape(-1, logits_shape[2]), torch.tensor(y).reshape(-1,))
            loss.backward()

            clip_grad_norm_(ner.parameters(), 5)
            optimizer.step()

    # Evaluation
    ner.eval()
    confusion_matrix = [[0, 0],
                        [0, 0]]
    test_data = NERDataset(args.test)
    for x, y in test_data:
        pred = torch.argmax(ner([x]), dim=-1).detach().numpy().reshape(-1,)
        y = np.array(y)
        tp = np.sum(pred[y == 1])
        tn = np.sum(1 - pred[y == 0])
        fp = np.sum(1 - y[pred == 1])
        fn = np.sum(y[pred == 0])

        confusion_matrix[0][0] += tn
        confusion_matrix[1][1] += tp
        confusion_matrix[0][1] += fp
        confusion_matrix[1][0] += fn

    table = [['', 'Predicted no name', 'Predicted name'],
             ['Real no name', confusion_matrix[0][0], confusion_matrix[0][1]],
             ['Real name', confusion_matrix[1][0], confusion_matrix[1][1]]]

    t = AsciiTable(table)
    print(t.table)
    print("Accuracy: {}".format(
        round((confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix), 4))
    )
