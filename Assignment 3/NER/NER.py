import mmap
import time
import numpy as np
import argparse
import sys
import codecs
from BinaryLogisticRegression import BinaryLogisticRegression

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class NER(object):
    """
    This class performs Named Entity Recognition (NER).

    It either builds a binary NER model (which distinguishes
    between 'name' or 'noname') from training data, or tries a NER model
    on test data, or both.

    Each line in the data files is supposed to have 2 fields:
    Token, Label

    The 'label' is 'O' if the token is not a name.
    """

    class Dataset(object):
        """
        Internal class for representing a dataset.
        """
        def __init__(self):

            #  The list of datapoints. Each datapoint is itself
            #  a list of features (each feature coded as a number).
            self.x = []

            #  The list of labels for each datapoint. The datapoints should
            #  have the same order as in the 'x' list above.
            self.y = []

    # --------------------------------------------------

    """
    Word vector feature computation
    """
    PAD_SYMBOL = "<pad>"
    MAX_CACHE_SIZE = 10000

    def mmap_read_word_vectors(fname):
        # Don't forget to close both when done with them
        file_obj = open(fname, mode="r", encoding="utf-8")
        mmap_obj = mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ)
        return mmap_obj, file_obj


    def capitalized_token(self):
        return self.current_token != None and self.current_token.istitle()


    def first_token_in_sentence(self):
        return self.last_token in [None, '.', '!', '?']


    def word_vector(self):
        if self.lowercase_fallback:
            token = self.current_token.lower()
        else:
            token = self.current_token
        
        if token != None:
            idx = self.w2id.get(token)
            if idx:
                return self.vec_cache[idx]
            else:

                p = self.pos_cache.get(token)
                if not p:
                    p = self.mo.find("\n{}".format(token).encode('utf8')) + 1
                    self.pos_cache[token] = p
                # normally find returns -1 if not found, but here we have +1
                if p > 0:
                    self.mo.seek(p)
                else:
                    v = self.w2id.get(NER.PAD_SYMBOL)
                    return self.vec_cache[v] if v else [-1] * self.D
        else:
            v = self.w2id.get(NER.PAD_SYMBOL)
            return self.vec_cache[v] if v else [-1] * self.D
        line = self.mo.readline()
        vec = list(map(float, line.decode('utf8').split()[1:]))

        if self.current_token_id < NER.MAX_CACHE_SIZE:
            self.w2id[token] = self.current_token_id
            self.vec_cache[self.current_token_id,:] = vec
            self.current_token_id += 1
        return vec


    class FeatureFunction(object):
        def __init__(self, func, boolean=True):
            self.func = func
            self.boolean = boolean

        def evaluate(self):
            if self.boolean:
                return 1 if self.func() else 0
            else:
                return self.func()



    # --------------------------------------------------

    def label_number(self, s):
        return 0 if 'O' == s else 1



    def read_and_process_data(self, filename):
        """
        Read the input file and return the dataset.
        """
        dataset = NER.Dataset()
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f.readlines():
                field = line.strip().split(',')
                if len(field) == 3:
                    # Special case: The token is a comma ","
                    self.process_data(dataset, ',', 'O')
                else:
                    self.process_data(dataset, field[0], field[1])
            return dataset
        return None



    def process_data(self, dataset, token, label):
        """
        Processes one line (= one datapoint) in the input file.
        """
        self.last_token = self.current_token
        self.current_token = token

        datapoint = []
        for f in self.features:
            res = f.evaluate()
            if type(res) == list or type(res) == np.ndarray:
                if datapoint:
                    datapoint.extend(res)
                else:
                    datapoint = res
            else:
                datapoint.append(res)
        dataset.x.append(datapoint)
        dataset.y.append(self.label_number(label))


    def read_model(self, filename):
        """
        Read a model from file
        """
        with codecs.open(filename, 'r', 'utf-8') as f:
            d = map(float, f.read().split(' '))
            return d
        return None

    # ----------------------------------------------------------


    def __init__(self, training_file, test_file, model_file, word_vectors_file,
                 stochastic_gradient_descent, minibatch_gradient_descent, lowercase_fallback):
        """
        Constructor. Trains and tests a NER model using binary logistic regression.
        """
        self.lowercase_fallback = lowercase_fallback

        self.current_token = None #  The token currently under consideration.
        self.last_token = None #  The token on the preceding line.

        # self.W, self.i2w, self.w2i = NER.read_word_vectors(word_vectors_file)
        self.mo, self.fo = NER.mmap_read_word_vectors(word_vectors_file)
        
        # get the dimensionality of the vectors
        p = self.mo.find("\nthe".encode('utf8')) + 1
        self.mo.seek(p)
        line = self.mo.readline()
        vec = list(map(float, line.decode('utf8').split()[1:]))
        self.D = len(vec)
        
        self.current_token_id = 0
        self.pos_cache, self.w2id, self.vec_cache = {}, {}, np.zeros((NER.MAX_CACHE_SIZE, self.D))
        self.w2id["the"] = self.current_token_id
        self.vec_cache[self.current_token_id,:] = vec
        self.current_token_id += 1

        p = self.mo.find(NER.PAD_SYMBOL.encode('utf8'))
        self.w2id[NER.PAD_SYMBOL] = self.current_token_id
        self.vec_cache[self.current_token_id,:] = vec
        self.current_token_id += 1

        # Here you can add your own features.
        self.features = [
            NER.FeatureFunction(self.word_vector, boolean=False),
            # NER.FeatureFunction(self.capitalized_token),
            # NER.FeatureFunction(self.first_token_in_sentence),
        ]

        if training_file:
            # Train a model
            training_set = self.read_and_process_data(training_file)
            if training_set:
                start_time = time.time()
                b = BinaryLogisticRegression(training_set.x, training_set.y)
                if stochastic_gradient_descent:
                    b.stochastic_fit_with_early_stopping()
                elif minibatch_gradient_descent:
                    b.minibatch_fit_with_early_stopping()
                else:
                    b.fit()
                print("Model training took {}s".format(round(time.time() - start_time, 2)))

        else:
            model = self.read_model(model_file)
            if model:
                b = BinaryLogisticRegression(model)


        # Test the model on a test set
        test_set = self.read_and_process_data(test_file)
        if test_set:
            b.classify_datapoints(test_set.x, test_set.y)


    # ----------------------------------------------------------

def main():
    """
    Main method. Decodes command-line arguments, and starts the Named Entity Recognition.
    """

    parser = argparse.ArgumentParser(description='Named Entity Recognition', usage='\n* If the -d and -t are both given, the program will train a model, and apply it to the test file. \n* If only -t and -m are given, the program will read the model from the model file, and apply it to the test file.')

    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-t', type=str,  required=True, help='test file (mandatory)')
    required_named.add_argument('-w', type=str, required=True, help='file with word vectors')

    group = required_named.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', type=str, help='training file (required if -m is not set)')
    group.add_argument('-m', type=str, help='model file (required if -d is not set)')

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('-s', action='store_true', default=False, help='Use stochastic gradient descent')
    group2.add_argument('-b', action='store_true', default=False, help='Use batch gradient descent')
    group2.add_argument('-mgd', action='store_true', default=False, help='Use mini-batch gradient descent')

    parser.add_argument('-lcf', '--lowercase-fallback', action='store_true')


    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
    arguments = parser.parse_args()

    ner = NER(arguments.d, arguments.t, arguments.m, arguments.w, arguments.s, arguments.mgd, arguments.lowercase_fallback)
    ner.mo.close()
    ner.fo.close()

    input("Press Return to finish the program...")


if __name__ == '__main__':
    main()
