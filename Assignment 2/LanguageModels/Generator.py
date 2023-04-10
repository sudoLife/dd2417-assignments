import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""


class Generator(object):
    """
    This class generates words from a language model.
    """

    def __init__(self):

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # FIXME: none of this is necessary for the generator
        # TODO: report this error?
        # # The average log-probability (= the estimation of the entropy) of the test corpus.
        # self.logProb = 0

        # # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        # self.last_index = -1

        # # The fraction of the probability mass given to unknown words.
        # self.lambda3 = 0.000001

        # # The fraction of the probability mass given to unigram probabilities.
        # self.lambda2 = 0.01 - self.lambda3

        # # The fraction of the probability mass given to bigram probabilities.
        # self.lambda1 = 0.99

        # # The number of words processed in the test corpus.
        # self.test_words_processed = 0

    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(
                    int, f.readline().strip().split(' '))
                # YOUR CODE HERE
                for i in range(self.unique_words):
                    line = f.readline().strip().split(' ')
                    token, count = line[1], int(line[2])
                    self.word[i] = token
                    self.index[token] = i
                    self.unigram_count[i] = count
                while True:
                    line = f.readline().strip().split(' ')
                    # EOF, essentially
                    if len(line) == 1:
                        break

                    first_index = int(line[0])
                    second_index = int(line[1])
                    log_prob = float(line[2])

                    self.bigram_prob[first_index][second_index] = log_prob

                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """
        print(w, end=' ')
        w = self.index[w]
        for i in range(n - 1):
            bigram = self.bigram_prob[w]
            w = random.choices(list(bigram.keys()), weights=[
                               math.exp(lp) for lp in bigram.values()], k=1)[0]
            print(self.word[w], end=' ')

        print("")


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Generator')
    parser.add_argument('--file', '-f', type=str,
                        required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str,
                        required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start, arguments.number_of_words)


if __name__ == "__main__":
    main()
