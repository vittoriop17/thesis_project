import os
import torch
from sklearn.model_selection import RandomizedSearchCV
import hdbscan
from sklearn.metrics import make_scorer
import umap
from sentence_transformers import SentenceTransformer
from typing import Literal
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

SEED = 42


def validity_index_score(estimator, X):
    try:
        print(f"X.shape: {X.shape}, estimator: {estimator}")
    except:
        pass
    y_pred = estimator.fit_predict(X)
    try:
        print(f"n clusters: {len(set(y_pred))}. N.outliers: {sum(y_pred==-1)}")
    except:
        pass
    return hdbscan.validity.validity_index(X, y_pred)


def get_sentences(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            if line.strip() != "":
                sentences.append(line)
    return sentences


class ClusteringPipeline:
    def __init__(self, bi_encoder_path, training_sentences: list = None,
                 metric=Literal['precomputed', 'cosine', 'euclidean'],
                 path_training_sentences=None, check_hopkins_test=False):
        # Set variables
        self.bi_encoder_path = bi_encoder_path
        self.metric = metric
        self.check_hopkins_test = check_hopkins_test
        self.hdbscan_model = None
        # if training_sentences is not empty, the variable path_training_sentences is not used
        self.training_sentences = training_sentences
        self.path_training_sentences = path_training_sentences

        # Check variables
        self._sanity_check()

        # Load fine-tuned bi-encoder model
        print(f"Loading pre-trained bi-encoder from path {self.bi_encoder_path}")
        self._load_model()

        # Load sentences if training_sentences is empty
        if training_sentences is None:
            print(f"Loading training sentences from {self.path_training_sentences}")
            self.training_sentences = get_sentences(self.path_training_sentences)

        # Extract embeddings and then compute similarity matrix (if necessary: metric=precomputed)
        self.training_embeddings, self.training_similarity_matrix = self._get_embeddings(self.training_sentences)

        # Check if embeddings are not uniformly distributed in the embedding space (hopkins test)
        self._check_hopkins()

        # Prepare HDBSCAN model
        self._prepare_evaluation()

    def _sanity_check(self):
        assert os.path.exists(
            self.bi_encoder_path), f"Invalid path for bi_encoder model. Passed value: {self.bi_encoder_path}"
        assert self.path_training_sentences is not None if self.training_sentences is None else True, \
            f"Must provide a path to training sentences if training sentences are not provided"
        assert os.path.exists(self.path_training_sentences) if self.training_sentences is None else True, \
            f"Invalid path for path_training_sentences. No file found ({self.path_training_sentences})"

    def _load_model(self):
        self.sbert_model = SentenceTransformer(self.bi_encoder_path,
                                               device="cuda" if torch.cuda.is_available() else "cpu")

    def _get_embeddings(self, sentences):
        sentence_embeddings = self.sbert_model.encode(sentences)
        cosine_similarity_matrix = cosine_similarity(sentence_embeddings) if self.metric == 'precomputed' else None
        # TODO - change UMAP arguments (like as n_neighbors and n_components). Add them as arguments to this class
        umap_model = umap.UMAP(metric='cosine', n_components=5)
        umap_sentence_embeddings = umap_model.fit_transform(sentence_embeddings)
        return umap_sentence_embeddings, cosine_similarity_matrix

    def _prepare_evaluation(self):
        self.param_dist = {'min_samples': [5, 10, 20, 30, 40, 50],
                           'min_cluster_size': [5, 10, 20, 25, 50, 100],
                           'cluster_selection_method': ['eom', 'leaf'],
                           'metric': ['euclidean']
                           }
        self.validity_scorer = make_scorer(hdbscan.validity.validity_index, greater_is_better=True)

    def _check_hopkins(self):
        if self.check_hopkins_test:
            raise NotImplementedError("Hopkins test not implemented yet")
            print("Evaluate 'cluster tendency' on training embeddings (after dimensionality reduction)"
                  "Cluster tendency consists to assess if clustering algorithms are relevant for a dataset.")
            # X = scale(self.training_embeddings)
            # hopkins(X, 150)

    def evaluate(self, evaluation_type):
        assert evaluation_type in ['dbcv', 'cvps'], f"Invalid argument for evaluation_type. Expected dbcv or cvps." \
                                                    f"Found: {evaluation_type}"
        if evaluation_type == 'dbcv':
            print(f"Starting evaluation with DBCV strategy")
            # TODO - add n_iter_search and param_dist (the keys of the dict) as class arguments
            n_iter_search = 10
            self.hdbscan_model = hdbscan.HDBSCAN(gen_min_span_tree=True).fit(self.training_embeddings.astype('double'))
            random_search = RandomizedSearchCV(self.hdbscan_model,
                                               param_distributions=self.param_dist,
                                               n_iter=n_iter_search,
                                               scoring=validity_index_score,
                                               random_state=SEED)
            random_search.fit(self.training_embeddings.astype('double'))
            print(f"Best Parameters {random_search.best_params_}")
            print(f"DBCV score :{random_search.best_estimator_.relative_validity_}")
        else:
            raise NotImplementedError("CVPS validation technique not implemented yet")
