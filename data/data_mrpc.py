"""
This module handles data coming from the MRPC dataset
"""
import os
import pandas as pd
import numpy as np
from silver_set_construction import *
from tqdm import tqdm


def get_all_sentences(data_path='MRPC\\msr_paraphrase_data.txt'):
    """
    :param data_path: str. Path to the dataset file.
        The dataset file must be stored in tsv format, with the following header:
            "sid", "sentence", "Author", "URL", "Agency", "Date", "Web_Date"
    :return: pandas dataframe. Sorted by web_date, containing sid and sentence information only.
    """
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}. Required valid filepath for MRPC data")
        exit(-1)
    df = pd.read_csv(data_path, skiprows=1, sep='\t', names=["sid", "sentence", "Author", "URL", "Agency", "Date", "Web_Date"])
    df['Web_Date'] = pd.to_datetime(df.Web_Date)
    df.sort_values(by="Web_Date", ascending=True, inplace=True)
    return df.loc[:, ["sid", "sentence"]]


def get_original_train_or_test_sentence_pairs(train=True):
    """
    Read all sentence pairs from training or test file.
    The function reads the ORIGINAL training/test file
    (original files downloaded from https://www.microsoft.com/en-us/download/details.aspx?id=52398)
    :param train: boolean.
    :return: pandas dataframe. Training/test sentence pairs. Each row of the dataset contains the following info:
        "score", "sid_1", "sid_2", "sentence_1", "sentence_2"
    """
    filepath = f"MRPC\\msr_paraphrase_{'train' if train else 'test'}.txt"
    df = pd.read_csv(filepath, skiprows=1, sep='\t', names=["score", "sid_1", "sid_2", "sentence_1", "sentence_2"])
    return df


def get_train_or_test_sentence_ids(train=True, df_sentence_pairs=None):
    df = get_original_train_or_test_sentence_pairs(train) if df_sentence_pairs is None else df_sentence_pairs
    sentence_id_set = set(df.sid_1).union(set(df.sid_2))
    return sentence_id_set


def remove_test_sentences(df__sif_sentence, test_sentence_ids):
    df_without_test_sentences = df__sif_sentence.loc[~df__sif_sentence['sid'].isin(test_sentence_ids)]
    return df_without_test_sentences


def get_validation_set(df_train=None, size=0.25, save=False):
    df_train = get_original_train_or_test_sentence_pairs(True) if df_train is None else df_train
    validation_set_len = int(len(df_train) * size)
    validation_set_idxs = np.random.choice(len(df_train), size=validation_set_len, replace=False)
    df_validation = df_train.iloc[validation_set_idxs]
    if save: df_validation[['score', 'sentence_1', 'sentence_2']].to_csv("MRPC/dev_set.tsv", sep="\t", header=True, index=False)
    return df_validation


def reformat_test_set():
    df_test = get_original_train_or_test_sentence_pairs(train=False)
    df_test = df_test[['score', 'sentence_1', 'sentence_2']]
    df_test.to_csv("MRPC/test_set.tsv", sep='\t', header=True, index=False)


if __name__=='__main__':
    df__sid_sentence = get_all_sentences()
    test_sentence_ids = get_train_or_test_sentence_ids(False)
    df__without_test_sentences = remove_test_sentences(df__sid_sentence, test_sentence_ids)
    df_validation = get_validation_set(save=True)
    reformat_test_set()
    val_sentence_ids = get_train_or_test_sentence_ids(False, df_validation)
    df__without_val_and_test_sentences = remove_test_sentences(df__without_test_sentences, val_sentence_ids)
    SilverSetConstructor(list(df__without_val_and_test_sentences.sentence), name='MRPC dataset',
                         verbose=False, folder="MRPC", task='classification')