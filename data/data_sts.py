"""
This module handles data coming from the STS dataset
"""
import os
import pandas as pd
import numpy as np
from silver_set_construction import *
from sklearn.preprocessing import KBinsDiscretizer


def get_test_data(validation=False):
    filepath = "STS\\sts-test.csv" if not validation else "STS\\sts-dev.csv"
    df = pd.read_csv(filepath, sep=",\t", names=["genre", "dataset", "year", "sid", "score", "sentence_1", "sentence_2", "useless1", "useless2"])
    df = df[["score", "sentence_1", "sentence_2"]]
    return df


def get_training_sentences():
    filepath = "STS\\sts-train.csv"
    df = pd.read_csv(filepath, sep=",\t", names=["genre", "dataset", "year", "sid", "score", "sentence_1", "sentence_2", "useless1", "useless2"])
    train_sentences = set(df.sentence_1).union(set(df.sentence_2))
    return train_sentences


def prepare_evaluation_data(dev=True):
    filepath = "STS\\sts-dev.csv" if dev else "STS\\sts-test.csv"
    out_filepath = "STS\\dev_set.tsv" if dev else "STS\\test_set.tsv"
    df = pd.read_csv(filepath, engine='python', sep=",\t", names=["genre", "dataset", "year", "sid", "score", "sentence_1", "sentence_2", "useless1", "useless2"])
    df = df[["score", "sentence_1", "sentence_2"]]
    df.score = df.score / 5
    df.to_csv(out_filepath, sep="\t", header=True, index=False)


def discretize_scores(df, bins=5, strategy='quantile'):
    """
    :param df: pandas DataFrame. Must contain the columns sentence_1, sentence_2, score
    :param bins: int, default=5. Number of bins (multiclass labels)
    :param strategy: string, {‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’. Check sklearn KBinsDiscretizer doc
    :return: pandas DataFrame and KBinsDiscretizer (after fitting).
     The pd dataframe contains discrete labels (0, 1, 2, ... bins-1) (stored under a new column discrete_score),
     and KBinsDiscretizer is the one obtained after fitting. It can be used to convert scores for dev and test sets.
    """
    kbins_discretizer = KBinsDiscretizer(n_bins=bins, strategy=strategy, encode='ordinal')
    df['discrete_score'] = kbins_discretizer.fit_transform(np.array(df.score).reshape((-1, 1)))
    return df, kbins_discretizer

if __name__=='__main__':
    # train_sentences = get_training_sentences()
    # SilverSetConstructor(list(train_sentences), name='STS dataset',
    #                      verbose=False, folder="STS", task='regression')
    test_set = get_test_data(False)
    discretize_scores(test_set)
    # prepare_evaluation_data(False)
    # prepare_evaluation_data(True)
