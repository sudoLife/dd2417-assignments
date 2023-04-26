import os
import time
import argparse
import string
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.

        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self._pad_word = '<pad>'
        self._sources = filenames
        self.emb_size = dimension
        self.lws = window_size
        self.rws = window_size
        self.context_size = self.lws + self.rws
        self.init_lr = learning_rate
        self.lr = learning_rate
        self.nsamples = nsample
        self.epochs = epochs
        self._nbrs = None
        self._use_corrected = use_corrected
        self._use_lr_scheduling = use_lr_scheduling
        self.estimator = None
        self.estimator_params = {'metric': 'cosine'}

    def init_params(self, W, index, word):
        self.focus = W
        self.index = index
        self.word = word

    @property
    def vocab_size(self):
        return len(self.word)

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list

        :param      line:  The line
        :type       line:  str
        """
        #
        # REPLACE WITH YOUR CODE HERE
        #
        translator = str.maketrans('', '', string.punctuation + string.digits)
        clean_line = line.translate(translator)
        words = clean_line.split()

        return words

    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self._sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)

    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices

        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        #
        # REPLACE WITH YOUR CODE
        #
        context_words = sent[max(i - self.lws, 0):i] + sent[i + 1:i + 1 + self.rws]
        return [self.index[word] for word in context_words]

    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        #
        # REPLACE WITH YOUR CODE
        #
        # Okay, so I'm assuming for a sentence it's like that:
        # we just slide the window over each sentence and create this thing
        # vocab = set()
        self.word = []
        self.index = {}
        self.unigram = []

        # for line in self.text_gen():
        #     for word in line:
        #         vocab.add(word)

        # self.word = list(vocab)
        # for i, word in enumerate(self.word):
        #     self.index[word] = i
        focus_words = []
        context_words = []

        # okay let's do it the efficient way, ig?
        for line in self.text_gen():
            for i, word in enumerate(line):
                if word not in self.index:
                    self.word.append(word)
                    self.index[word] = len(self.word) - 1
                    self.unigram.append(1)
                else:
                    self.unigram[self.index[word]] += 1

            # I am going through a second time because I want to use indices
            # instead of words themselves
            for i, word in enumerate(line):
                focus_words.append(self.index[word])
                # such a useless method tbh
                context_words.append(self.get_context(line, i))

        unigram_sum = sum(self.unigram)
        self.unigram = np.asarray(self.unigram, dtype=float) / unigram_sum
        self.unigram_corrected = self.unigram ** 0.75
        self.unigram_corrected *= 1/np.sum(self.unigram_corrected)
        self.ns_rng = np.random.default_rng(seed=42)
        self.train_rng = np.random.default_rng(seed=420)

        return focus_words, context_words

    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def negative_sampling(self, number: int, xb: int, pos: int):
        """Generate negative samples

        Args:
            number (int): number of negative samples to be generated
            xb (int): index of the current focus word
            pos (int): index of the current positive example

        Returns:
            list[int]: negative sampled words
        """
        #
        # REPLACE WITH YOUR CODE
        #
        samples = np.zeros((number, self.emb_size))

        i = 0
        while i < number:
            index = self.index[self.ns_rng.choice(self.word, p=self.unigram_corrected)]

            if index in (xb, pos):
                continue

            samples[i] = index
            i += 1

        return samples

    def negative_sampling_batch(self, number: int, focus_word: int, context: list) -> list[int]:
        samples = np.zeros((len(context) * number), dtype=int)

        i = 0
        while i < number:
            # TODO: honor use_corrected option
            index = self.index[self.ns_rng.choice(self.word, p=self.unigram_corrected)]

            if index == focus_word or index in context:
                continue

            samples[i] = index
            i += 1

        return samples

    def calculate_learning_rate(self, processed_words: int):
        if not self._use_lr_scheduling:
            return

        if self.lr < self.init_lr * 1e-4:
            self.lr = self.init_lr * 1e-4
        else:
            self.lr = self.init_lr * (1 - processed_words / (self.epochs * self.vocab_size + 1))

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        focus_words, context_words = self.skipgram_data()
        N = len(focus_words)
        print("Dataset contains {} datapoints".format(N))

        # REPLACE WITH YOUR RANDOM INITIALIZATION
        self.focus = self.train_rng.normal(loc=0.0, scale=0.01, size=(N, self.emb_size))
        self.context = self.train_rng.normal(loc=0.0, scale=0.01, size=(N, self.emb_size))

        for ep in range(self.epochs):
            # TODO: should I shuffle?
            for i in tqdm(range(N)):
                #
                # YOUR CODE HERE
                #
                word = focus_words[i]
                positive_examples = context_words[i]
                if len(positive_examples) == 0:
                    continue
                negative_examples = self.negative_sampling_batch(self.nsamples, word, positive_examples)
                # positive examples
                focus_grad = np.zeros(self.emb_size)
                # loss = 0.0
                pos_contexts = self.context[positive_examples]
                neg_contexts = self.context[negative_examples]

                pos_activations = self.sigmoid(pos_contexts @ self.focus[word]).reshape(-1, 1) - 1
                pos_context_grad = pos_activations * self.focus[word]
                # FIXME: this pos_contexts is old, but maybe that's okay?
                focus_grad += np.mean(pos_activations * pos_contexts, axis=0)
                self.context[positive_examples] -= self.lr * pos_context_grad

                neg_activations = self.sigmoid(neg_contexts @ self.focus[word]).reshape(-1, 1)
                neg_context_grad = neg_activations * self.focus[word]

                focus_grad += np.mean(neg_activations * neg_contexts, axis=0)
                self.context[negative_examples] -= self.lr * neg_context_grad

                self.focus[word] -= self.lr * focus_grad

    def find_nearest(self, words, k=5, metric='cosine'):
        """
        Function returning k nearest neighbors with distances for each word in `words`

        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.

        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):

        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]

        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.

        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        #
        # REPLACE WITH YOUR CODE
        #
        estimator_params = {'metric': metric}
        if self.estimator is None or self.current_estimator_params != estimator_params:
            # retrain
            self.estimator_params = estimator_params

            self.estimator = NearestNeighbors(**self.estimator_params, n_jobs=-1)
            self.estimator.fit(self.focus)

        words = [self.index[word] for word in words]
        word_embeddings = self.focus[words]
        kneighbors = self.estimator.kneighbors(word_embeddings, n_neighbors=k, return_distance=True)

        results = []
        for i in range(len(words)):
            result = []
            distances, neighbors = kneighbors[0][i], kneighbors[1][i]

            for distance, neighbor in zip(distances, neighbors):
                result.append((self.word[neighbor], round(distance, 2)))

            results.append(result)

        return results

    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", 'w') as f:
                W = self.focus
                f.write("{} {}\n".format(self.vocab_size, self.emb_size))
                for i, w in enumerate(self.word):
                    f.write(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i, :])) + "\n")
        except Exception as e:
            print("Error: failing to write model to the file")

    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v

    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text)

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')

    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument(
        '-t', '--text',
        default='/home/sudolife/Documents/KTH/Language Engineering/Assignment 3/word2vec/harry_potter_1.txt',
        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=50, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    # parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
