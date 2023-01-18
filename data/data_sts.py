"""
This module handles data coming from the STS dataset
"""
import os
import pandas as pd
import numpy as np
from silver_set_construction import *


def get_test_data(validation=False):
    filepath = "STS\\sts-test.csv" if not validation else "STS\\sts-dev.csv"
    df = pd.read_csv(filepath, sep=",\t", names=["genre", "dataset", "year", "sid", "score", "sentence_1", "sentence_2"])
    df = df[["score", "sentence_1", "sentence_2"]]
    return df


def get_training_sentences():
    filepath = "STS\\sts-train.csv"
    df = pd.read_csv(filepath, sep=",\t", names=["genre", "dataset", "year", "sid", "score", "sentence_1", "sentence_2"])
    train_sentences = set(df.sentence_1).union(set(df.sentence_2))
    return train_sentences


def prepare_evaluation_data(dev=True):
    filepath = "STS\\sts-dev.csv" if dev else "STS\\sts-test.csv"
    out_filepath = "STS\\dev_set.tsv" if dev else "STS\\test_set.tsv"
    df = pd.read_csv(filepath, engine='python', sep=",\t", names=["genre", "dataset", "year", "sid", "score", "sentence_1", "sentence_2", "useless1", "useless2"])
    df = df[["score", "sentence_1", "sentence_2"]]
    df.score = df.score / 5
    df.to_csv(out_filepath, sep="\t", header=True, index=False)


if __name__=='__main__':
    train_sentences = get_training_sentences()
    SilverSetConstructor(list(train_sentences), name='STS dataset',
                         verbose=False, folder="STS", task='regression')
    # prepare_evaluation_data(False)
    # prepare_evaluation_data(True)
