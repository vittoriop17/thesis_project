"""
This module handles the creation of the silver set.
It is based on SIF-controlled extraction and labeling.
Input needed: set of sentences_wo_punct, type of task (regression or classification).
    These may vary based on the used dataset  (MRPC: classification, STS: regression)
"""
import pandas as pd
from models import sentence_embedding as se
from models import pipeline
import numpy as np
from utils.utils import cosine_similarity, euclidean_similarity
import os
import time
import string


N_RANDOM_SENTENCES = 3_000
N_RANDOM_PAIRS = 1_000
SIF_CONTROLLED_PAIRS_PER_SENTENCE = 2
# total number of sentence pairs:
#   N_RANDOM_PAIRS + [SIF_CONTROLLED_PAIRS_PER_SENTENCE * (N_RANDOM_SENTENCES - N_RANDOM_PAIRS)] = 5_000
#       1_000      + [              2                   * (         3_000     -         1_000 )] = 5_000


class SilverSetConstructor:
    def __init__(self, sentences, name, folder, verbose=True, thr=0.5, task='regression', similarity_function='cosine',
                 seed=42, filepath=None, n_pairs=None):
        np.random.seed(seed)
        assert task in ['regression', 'classification'],\
            f"Invalid value for 'task' argument. Expected 'regression' or 'classification'. Found {task} instead"
        assert similarity_function in ['cosine', 'euclidean'],\
            f"Invalid value for similarity_function argument. Expected 'cosine' or 'euclidean'. Found {similarity_function} instead"
        assert 0<thr<1, f"Invalid value for 'thr' (threshold). Expected value between 0 and 1 (excluded). Found {thr}"
        # Remove punctuation from sentences_wo_punct!
        # N.B. this is need only before applying SIF. No need to remove punct for bert
        self.n_pairs = n_pairs
        self.sentences_wo_punct = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences]
        self.sentences_w_punct = sentences
        self.verbose = verbose
        self.task = task
        self.thr = thr
        self.similarity_function = similarity_function
        self.pairs = []  # list of tuples: (sentence_1, sentence_2)
        self.labeled_pairs = []  # list of tuples: ((sentence_1, sentence_2), score)
        # instantiate the usif model (used for extraction and labeling)
        print(f"Start Silver Set Construction - {name}."
              f"Total training sentences_wo_punct: {len(self.sentences_wo_punct)}\n")
        start_time = time.time()
        self.usif_model = se.SifEmbedding(self.sentences_wo_punct)
        self.extract_sentence_pairs()
        self.label_sentence_pairs()
        print(f"End Silver Set Construction."
              f"Execution time: {time.time() - start_time} sec")
        filepath = os.path.join(folder, filepath) if filepath is not None else os.path.join(folder, f"silver_set_{task}_{similarity_function}.tsv") if task == 'regression' else \
            os.path.join(folder, f"silver_set_{task}{thr}_{similarity_function}.tsv")
        np.random.shuffle(self.labeled_pairs)
        self.save_sentence_pairs(filepath)

    def extract_sentence_pairs(self):
        # N.B.: the pairs are made using the sentences WITH punctuation!
        # Because BERT model does not require to remove punctuation.
        # It is actually better to keep it
        N_SENTENCES = len(self.sentences_wo_punct)
        sentence_ids = np.random.choice(N_SENTENCES, size=N_RANDOM_SENTENCES, replace=False)
        sentence_ids_for_random_pairs = sentence_ids[:N_RANDOM_PAIRS]
        sentence_ids_for_sif_controlled_pairs = sentence_ids[N_RANDOM_PAIRS:]
        self.pairs.extend([(self.sentences_w_punct[id], self.sentences_w_punct[np.random.choice(N_SENTENCES, 1)[0]]) for id in sentence_ids_for_random_pairs])
        for idx in sentence_ids_for_sif_controlled_pairs:
            s1 = self.sentences_w_punct[idx]
            most_similar = self.usif_model.most_similar_by_sentence_idx(idx, SIF_CONTROLLED_PAIRS_PER_SENTENCE)
            if self.verbose:
                print(f"S1: {s1}\tS2: {most_similar[0][0]}, \tScore: {most_similar[0][2]}")
            self.pairs.extend([(s1, s2[0]) for s2 in most_similar])
        if self.n_pairs:
            # self.pairs = np.random.choice(self.pairs, self.n_pairs, replace=False)
            self.pairs = np.array(self.pairs)[np.random.choice(len(self.pairs), self.n_pairs, replace=False)]

    def label_sentence_pairs(self):
        self.labeled_pairs.extend([(pair, self.score(pair)) for pair in self.pairs])

    def score(self, pair):
        similarity = cosine_similarity if self.similarity_function == 'cosine' else euclidean_similarity
        score = similarity(self.usif_model.embeddings([pair[0]])[0], self.usif_model.embeddings([pair[1]])[0])
        score = score if self.task == 'regression' else (1 if score > self.thr else 0)
        return score

    def save_sentence_pairs(self, filepath):
        with open(filepath, "w", encoding='utf-8') as fout:
            fout.write("sentence_1\tsentence_2\tscore\n")
            for labeled_pair in self.labeled_pairs:
                score = np.abs(labeled_pair[1])
                score = 1 if score > 1 else score
                fout.write(f"{labeled_pair[0][0]}\t{labeled_pair[0][1]}\t{score}\n")
