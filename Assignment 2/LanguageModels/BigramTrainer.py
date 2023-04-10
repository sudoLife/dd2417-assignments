#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs

"""
This file is part of the computer assignments for the course DD2417 Language Engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file @code{f}.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = str(text_file.read()).lower()
        try:
            # Important that it is named self.tokens for the --check flag to work
            self.tokens = nltk.word_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)

    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """
        # YOUR CODE HERE
        self.total_words += 1

        index = None
        if token not in self.index:
            # new unique token, add it
            self.index[token] = self.unique_words
            self.word[self.unique_words] = token
            index = self.unique_words
            self.unique_words += 1
        else:
            index = self.index[token]

        self.unigram_count[index] += 1
        # TODO: find out if we need the starter sentence, a.k.a how probable it is that a sentence starts with current token
        if self.last_index != -1:
            self.bigram_count[self.last_index][index] += 1
        # the very last operation
        self.last_index = index

    def stats(self):
        """
        Creates a list of rows to print of the language model.

        """
        rows_to_print = []

        # YOUR CODE HERE
        rows_to_print.append(f'{self.unique_words} {self.total_words}')
        # unigrams
        for i, token in self.word.items():
            rows_to_print.append(f'{i} {token} {self.unigram_count[i]}')
        # bigrams
        for first_index, second_dict in self.bigram_count.items():
            for second_index, count in second_dict.items():
                log_prob = math.log(count / self.unigram_count[first_index])

                rows_to_print.append(
                    f'{first_index} {second_index} {log_prob:.15f}')

        rows_to_print.append('-1')
        return rows_to_print

    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        self.laplace_smoothing = False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True,
                        help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str,
                        help='file in which to store the language model')

    arguments = parser.parse_args()

    bigram_trainer = BigramTrainer()

    bigram_trainer.process_files(arguments.file)

    stats = bigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8') as f:
            for row in stats:
                f.write(row + '\n')
    else:
        for row in stats:
            print(row)


if __name__ == "__main__":
    main()
