import os
import pickle
from parse_dataset import Dataset
from dep_parser import Parser
from logreg import LogisticRegression
import numpy as np


class TreeConstructor:
    """
    This class builds dependency trees and evaluates using unlabeled arc score (UAS) and sentence-level accuracy
    """

    def __init__(self, parser: Parser):
        self.__parser = parser

    def build(self, model, words, tags, ds):
        """
        Build the dependency tree using the logistic regression model `model` for the sentence containing
        `words` pos-tagged by `tags`

        :param      model:  The logistic regression model
        :param      words:  The words of the sentence
        :param      tags:   The POS-tags for the words of the sentence
        :param      ds:     Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #
        # In the 'build' method, you need to take the current configuration (stack,
        # buffer, tree), and turn that into a data point from which the model can
        # predict the next action. Then, you perform that action, get a new
        # configuration, turn that into a data point, predict the next action, and so
        # on. Keep going until you reach an end state with an empty buffer and a stack
        # containing only ROOT.
        parser = self.__parser

        i, stack, pred_tree = 0, [], [0] * len(words)
        move_log = []

        while i < len(pred_tree) or len(stack) > 1:
            # get the prediction and move stuff
            datapoint = ds.dp2array(words, tags, i, stack)
            # expand the dims
            prediction = model.get_log_probs(datapoint)
            moves = np.argsort(prediction)[::-1]

            valid_moves = parser.valid_moves(i, stack, pred_tree)
            selected_move = -1
            for move in moves:
                if move in valid_moves:
                    selected_move = move
                    break
            # NOTE: I'm assuming at least one move is valid?
            i, stack, pred_tree = parser.move(
                i, stack, pred_tree, selected_move)
            move_log.append(selected_move)
        return move_log

    def evaluate(self, model, test_file, ds):
        """
        Evaluate the model on the test file `test_file` using the feature representation given by the dataset `ds`

        :param      model:      The model to be evaluated
        :param      test_file:  The CONLL-U test file
        :param      ds:         Training dataset instance having the feature maps
        """

        # YOUR CODE HERE

        parser = self.__parser
        # test_ds = parser.create_dataset(test_file)
        correct_sentences = 0
        total_sentences = 0
        uas_total = 0
        uas_correct = 0
        with open(test_file, 'r') as source:
            for words, tags, tree, relations in parser.trees(source):
                # call the build function
                correct_moves = parser.compute_correct_moves(tree)
                moves = self.build(model, words, tags, ds)

                total_sentences += 1
                # correct_sentences += moves == correct_moves
                is_correct = True
                for i in range(len(correct_moves)):
                    # NOTE: will they ever be different, I wonder?
                    if correct_moves[i] != parser.SH:
                        uas_total += 1

                        if correct_moves[i] == moves[i]:
                            uas_correct += 1
                    if correct_moves[i] != moves[i]:
                        is_correct = False
                correct_sentences += is_correct
            print(
                f'Sentence-level accuracy: {correct_sentences / total_sentences}')
            print(f'UAS accuracy: {uas_correct / uas_total}')


train_set_path = "/home/sudolife/Documents/KTH/Language Engineering/Assignment 2/DepParser/en-ud-train-projective.conllu"
dev_set_path = "/home/sudolife/Documents/KTH/Language Engineering/Assignment 2/DepParser/en-ud-dev-projective.conllu"
model_path = "/home/sudolife/Documents/KTH/Language Engineering/Assignment 2/DepParser/model.pkl"


def main():
    # Create parser
    p = Parser()

    # Create training dataset
    ds = p.create_dataset(train_set_path, train=True)

    # Train LR model
    if os.path.exists(model_path):
        # if model exists, load from file
        print("Loading existing model...")
        mlr = pickle.load(open(model_path, 'rb'))
    else:
        # train model using minibatch GD
        mlr = LogisticRegression()
        mlr.fit(*ds.to_arrays())
        pickle.dump(mlr, open(model_path, 'wb'))

    # Create test dataset
    test_ds = p.create_dataset(dev_set_path, train=False)
    # Copy feature maps to ensure that test datapoints are encoded in the same way
    test_ds.copy_feature_maps(ds)
    # Compute move-level accuracy
    mlr.classify_datapoints(*test_ds.to_arrays())

    # Compute UAS and sentence-level accuracy
    t = TreeConstructor(p)
    t.evaluate(mlr, dev_set_path, ds)


if __name__ == '__main__':
    main()
