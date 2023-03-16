import json
import os
import torch
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
import hdbscan
from sklearn.metrics import make_scorer
import umap
from umap import validation
from sentence_transformers import SentenceTransformer
from typing import Literal
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from joblib import Memory
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
import joblib
import pandas as pd
import plotly.express as px

LOCATION = '/tmp/joblib'
SEED = 42


def validity_index_score(estimator, X_test):
    # work-around
    X_train = estimator.prediction_data_.raw_data if estimator.metric != 'precomputed' else X_test
    # try:
    #     print(f"X.shape: {X_test.shape}, estimator: {estimator}")
    #     print(f"X_train.shape: {X_train.shape}")
    # except:
    #     pass
    # y_pred = estimator.fit_predict(X_test)
    # extract the labels predicted for the training set
    y_pred_training = estimator.labels_
    try:
        print(f"\n\nEstimator (metric={estimator.metric}:"
              f"\n{estimator}"
              f"\nN.clusters: {len(set(y_pred_training))}. N.outliers: {sum(y_pred_training == -1)}")
    except:
        pass
    return hdbscan.validity.validity_index(X_train, y_pred_training, metric=estimator.metric)


def get_clustering_results(folder):
    data = []  # list of dicts
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath) and filename != '.gitignore':
            data.append(json.load(open(filepath, "r")))
    print(f"Total configurations: {len(data)}")
    return pd.DataFrame.from_records(data)


def get_sentences(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            if line.strip() != "":
                sentences.append(line)
    return sentences


def plot_analysis(df):
    top_10_scores = set(sorted(df.score)[::-1][:10])
    df[df.score.isin(top_10_scores)]


class ClusteringPipeline:
    def __init__(self, bi_encoder_path, training_sentences: list = None,
                 train_sentences_path=None, check_hopkins_test=False,
                 validate_umap=False, n_components=5, umap_min_dist=0.1, umap_n_neighbors=15, umap_metric='cosine',
                 hdbscan_min_samples=5, hdbscan_min_cluster_size=5, hdbscan_metric='euclidean',
                 hdbscan_cluster_method='eom',
                 hdbscan_epsilon=0.2, folder_results=os.path.join("results", "HDBSCAN"),
                 path_df_sentences_and_embeddings=None, **kwargs):
        # Set variables
        self.bi_encoder_path = bi_encoder_path
        self.n_components = n_components
        self.check_hopkins_test = check_hopkins_test
        self.validate_umap = validate_umap
        self.hdbscan_model = None
        self.original_sentence_embeddings = None
        self.folder_results = folder_results
        self.umap_params = {'metric': umap_metric,
                            'n_components': self.n_components,
                            'min_dist': umap_min_dist,
                            'n_neighbors': umap_n_neighbors}
        self.hdbscan_params = {'min_samples': hdbscan_min_samples,
                               'min_cluster_size': hdbscan_min_cluster_size,
                               'metric': hdbscan_metric,
                               'cluster_selection_method': hdbscan_cluster_method,
                               'cluster_selection_epsilon': hdbscan_epsilon}
        # if training_sentences is not empty, the variable path_training_sentences is not used
        self.training_sentences = training_sentences
        self.path_training_sentences = train_sentences_path

        # Check variables
        self._sanity_check()

        # Load existing results from folder_results (only when parameter search is applied for hdbscan)
        self._load_partial_results()

        # Load fine-tuned bi-encoder model
        print(f"Loading pre-trained bi-encoder from path {self.bi_encoder_path}")
        self._load_model()

        # Load sentences if training_sentences is empty
        if training_sentences is None:
            if path_df_sentences_and_embeddings is None:
                print(f"Loading training sentences from {self.path_training_sentences}")
                self.training_sentences = get_sentences(self.path_training_sentences)
                # self.training_sentences = self.training_sentences[:100]
            else:
                print(f"Loading training sentences and training embeddings from {path_df_sentences_and_embeddings}")
                df_s_e = pd.read_csv(path_df_sentences_and_embeddings)
                self.training_sentences = df_s_e['Unnamed: 0']
                self.original_training_embeddings = df_s_e[[f'{i}' for i in range(768)]]

        # Extract embeddings and then compute similarity matrix (if necessary: metric=precomputed)
        self.training_embeddings = self._get_embeddings(self.training_sentences)
        self.training_embeddings = self.training_embeddings.astype('double')
        # Check if embeddings are not uniformly distributed in the embedding space (hopkins test)
        self._check_hopkins()

    def _sanity_check(self):
        assert os.path.exists(self.bi_encoder_path) or self.bi_encoder_path=='bert-base-uncased',\
            f"Invalid path for bi_encoder model. Passed value: {self.bi_encoder_path}"
        assert self.path_training_sentences is not None if self.training_sentences is None else True, \
            f"Must provide a path to training sentences if training sentences are not provided"
        assert os.path.exists(self.path_training_sentences) if self.training_sentences is None else True, \
            f"Invalid path for path_training_sentences. No file found ({self.path_training_sentences})"
        if os.path.basename(os.getcwd()) == "models":
            self.folder_results = os.path.join("..", self.folder_results)
        assert os.path.exists(self.folder_results), f"Result folder not found: path: {self.folder_results}"

    def _load_model(self):
        self.sbert_model = SentenceTransformer(self.bi_encoder_path,
                                               device="cuda" if torch.cuda.is_available() else "cpu")

    def _get_embeddings(self, sentences):
        if self.original_training_embeddings is None:
            sentence_embeddings = self.sbert_model.encode(sentences)
            self.original_sentence_embeddings = sentence_embeddings
        # app = np.concatenate([np.array(sentences).reshape(-1, 1),
        #                       self.original_sentence_embeddings.reshape(len(sentences), -1)
        #                       ], axis=-1)
        # cols = ['sentence']
        # cols.extend([f"x{i}" for i in range(self.original_sentence_embeddings.shape[-1])])
        # pd.DataFrame(app, columns=cols).to_csv("sentence_embeddings_no_fine_tuning.csv", index=False)
        if self.validate_umap:
            self._validate_umap(self.original_sentence_embeddings)
        umap_model = umap.UMAP(**self.umap_params)
        umap_sentence_embeddings = umap_model.fit_transform(self.original_sentence_embeddings)
        self.umap_model = umap_model
        return umap_sentence_embeddings

    def _load_partial_results(self):
        self.existing_ids = []
        for filename in os.listdir(self.folder_results):
            try:
                mid = int(filename.strip())
                self.existing_ids.append(mid)
            except:
                pass
        print(f"Found {len(self.existing_ids)} experiments")

    def _validate_umap(self, sentence_embeddings):
        print(f"Starting UMAP hyperparameters tuning (based on trustworthiness metric)...")
        min_dists = (0.05, 0.2, 0.5)
        n_components = [5, 10, 15]
        metrics = ['cosine']
        params_and_trust_values = list()
        for min_dist in min_dists:
            for n_c in n_components:
                for metric in metrics:
                    umap_model = umap.UMAP(metric=metric, n_components=n_c, min_dist=min_dist)
                    umap_sentence_embeddings = umap_model.fit_transform(sentence_embeddings)
                    trustworthiness = validation.trustworthiness_vector(source=sentence_embeddings.astype('double'),
                                                                        embedding=umap_sentence_embeddings.astype(
                                                                            'double'), max_k=20)
                    print(f"Model params:"
                          f"\tmin_dist: {min_dist},\tn_components: {n_c},\tmetric: {metric}\ttrustworthiness: {trustworthiness[15]}")
                    params_and_trust = {'min_dist': min_dist,
                                        'n_components': n_c,
                                        'trustworthiness': trustworthiness[15]}
                    params_and_trust_values.append(params_and_trust)
        best_params = params_and_trust_values[
            np.argmax(list(map(lambda x: x['trustworthiness'], params_and_trust_values)))]
        print(f"Best UMAP parameters and trustworthiness: "
              f"{json.dumps(best_params, indent=4)}")
        for k, v in best_params.items():
            if k != 'trustworthiness':
                self.umap_params[k] = v

    def _prepare_evaluation(self, metric):
        self.param_dist = {'min_samples': [1, 2, 5, 10, 25, 50], 'min_cluster_size': [5, 10, 25, 50],
                           'cluster_selection_method': ['eom', 'leaf'], 'metric': [metric],
                           'cluster_selection_epsilon': [0.05, 0.1, 0.2],
                           'prediction_data': [True] if metric == 'euclidean' else [False], 'gen_min_span_tree': [True]}
        self.cosine_similarity_matrix = cosine_similarity(self.training_embeddings)

    def _check_hopkins(self):
        if self.check_hopkins_test:
            raise NotImplementedError("Hopkins test not implemented yet")
            # print("Evaluate 'cluster tendency' on training embeddings (after dimensionality reduction)"
            #       "Cluster tendency consists to assess if clustering algorithms are relevant for a dataset.")
            # X = scale(self.training_embeddings)
            # hopkins(X, 150)

    def evaluate(self):
        score_euclidean = self._evaluate_metric("euclidean")
        # score_euclidean = 0
        best_params_euclidean = self.hdbscan_params
        best_hdbscan_model = self.hdbscan_model
        score_precomputed = self._evaluate_metric("precomputed")
        best_metric = 'precomputed'
        if score_euclidean > score_precomputed:
            self.hdbscan_params = best_params_euclidean
            self.hdbscan_model = best_hdbscan_model
            best_metric = 'euclidean'
        print(f"Best metric: {best_metric}")
        print(f"Final set of parameters: "
              f"\n\t{json.dumps(self.hdbscan_params, indent=4)}")

    def _evaluate_metric(self, metric):
        metric = str.lower(metric)
        assert metric in ['euclidean', 'precomputed'], f"Invalid metric! Must be euclidean or precomputed"
        print(f"Starting evaluation with DBCV strategy. Using {str(metric).upper()} as distance metric\n")
        self._prepare_evaluation(metric)
        X = self.training_embeddings if metric == 'euclidean' else self.cosine_similarity_matrix
        n_iter_search = 40
        # hdbscan_model = hdbscan.HDBSCAN(gen_min_span_tree=True, prediction_data=True, metric=metric)
        best_validitiy_score = - np.inf
        best_params = {}
        param_list = list(ParameterSampler(param_distributions=self.param_dist, n_iter=n_iter_search))
        for params in tqdm(param_list):
            # params['memory'] = Memory(LOCATION, verbose=0)  # Speed up computation
            hdbscan_model = hdbscan.HDBSCAN(**params)
            hdbscan_model.fit(X.astype('double'))
            mid = int(hash(frozenset(params.items())))
            # check if the current set of parameters has been already used
            if mid in self.existing_ids:
                print(f"\033[93mSet of params already tested\n\33[30m\n")
                continue
            print("\n---------------------------------------------------------\n")
            print(f"\nExperiment with params: {json.dumps(params, indent=2)}"
                  f"\nModel: {hdbscan_model}\n")
            score = hdbscan.validity.validity_index(X, hdbscan_model.labels_, metric=metric,
                                                    d=self.umap_params['n_components'])
            print(f"\033[94mScore {score}"
                  f"\nN.Clusters: {len(set(hdbscan_model.labels_))}"
                  f"\nOutliers: {sum(hdbscan_model.labels_ == -1)}\n\33[30m")
            if score > best_validitiy_score:
                best_validitiy_score = score
                best_params = params
            params['n_clusters'] = len(set(hdbscan_model.labels_))
            params['outliers'] = int(sum(hdbscan_model.labels_ == -1))
            params['score'] = score
            self.save_partial_results_hdbscan(params, mid)
            del hdbscan_model
        best_params.pop('memory', None)
        print(f"\nDBCV score :{best_validitiy_score}")
        print(f"\nBest Parameters {best_params}")
        print(f"\nOverriding existing params with best params:"
              f"\n\nExisting params: \n\t{json.dumps(self.hdbscan_params, indent=4)}"
              f"\n\nNew params (best params after random grid search): \n\t{json.dumps(best_params, indent=4)}")
        best_params.pop('n_clusters', None)
        best_params.pop('outliers', None)
        best_params.pop('score', None)
        self.hdbscan_params = best_params
        self.hdbscan_model = None
        return best_validitiy_score

    def train_over_all_sentences(self):
        X = cosine_similarity(self.training_embeddings) if self.hdbscan_params['metric'] == 'precomputed' \
            else self.training_embeddings
        if self.hdbscan_model is None:
            self.hdbscan_model = hdbscan.HDBSCAN(prediction_data=True, **self.hdbscan_params)
        self.hdbscan_model.fit(X)
        # test_labels, _ = hdbscan.approximate_predict(self.hdbscan_model, self.test_sentences)
        # Drawbacks
        # The Calinski-Harabasz AND davies_bouldin_score index is generally higher for convex clusters than other concepts of clusters,
        # such as density based clusters like those obtained through DBSCAN.
        print(f"Evaluation on training data:"
              f"\t(calinski_harabasz_score): {calinski_harabasz_score(self.training_embeddings, self.hdbscan_model.labels_)}"
              f"\t(davies_bouldin_score): {davies_bouldin_score(self.training_embeddings, self.hdbscan_model.labels_)}"
              f"\tValidity index: {hdbscan.validity.validity_index(self.training_embeddings, self.hdbscan_model.labels_)}"
              f"\n\n"

              f"Number of clusters: {len(set(self.hdbscan_model.labels_))}\n"
              f"Number of outliers: {sum(self.hdbscan_model.labels_ == -1)}"
              f"\n\n"

              f"Evaluation on test data: TODO"
              # f"\t(calinski_harabasz_score): {calinski_harabasz_score(self.test_embeddings, test_labels)}"
              # f"\t(davies_bouldin_score): {davies_bouldin_score(self.test_embeddings, test_labels)}"
              # f"\tValidity index: {hdbscan.validity.validity_index(self.test_embeddings, test_labels)}"

              f"\n\n"
              f"Note: \n"
              f"\tcalinski_harabasz_score: the larger the better.\n"
              f"\tdavies_bouldin_score: the lower the better.")

    def save_partial_results_hdbscan(self, hdbscan_params_and_results, mid):
        filepath = os.path.join(self.folder_results, str(mid))
        json.dump(hdbscan_params_and_results, open(filepath, "w"), indent=4, sort_keys=True)

    def plot_clusters(self, sentence_embeddings=None):
        # N.b.: the HDBSCAN algorithm is applied to the sentence embedding in a N-dimensional space.
        # then, the cluster information is used to plot the datapoints in a 2-dimensional space.
        # thus, be aware that the data for HDBSCAN and the data used for visualization lie in two different spaces!
        sentence_embeddings = self.original_sentence_embeddings if sentence_embeddings is None else sentence_embeddings
        umap_params = self.umap_params
        umap_params['n_components'] = 2
        bi_dim_umap_model = umap.UMAP(**umap_params)
        bidim_sentence_embeddings = bi_dim_umap_model.fit_transform(sentence_embeddings)
        sentence_embeddings_for_clustering = self.umap_model.transform(sentence_embeddings)
        predictions, probs = hdbscan.approximate_predict(self.hdbscan_model, sentence_embeddings_for_clustering)
        print(predictions)
        predictions = np.array(predictions).astype('int') + 1
        self.plot_cluster_dist(predictions)
        n_clusters = len(set(predictions))
        colors = np.array([list(np.random.choice(range(256), size=3)) for _ in range(n_clusters)]) / 255
        colors[0] = (1, 0, 0)
        plt.scatter(x=bidim_sentence_embeddings[:, 0], y=bidim_sentence_embeddings[:, 1],
                    alpha=0.5, c=colors[predictions], s=1)
        plt.savefig("clustering.png", bbox_inches='tight')
        plt.show()
        predictions_wo_outliers = predictions[predictions != 0]
        colors_wo_outliers = colors[predictions_wo_outliers]
        bidim_sentence_embeddings_wo_outlies = bidim_sentence_embeddings[predictions != 0, :]
        plt.scatter(x=bidim_sentence_embeddings_wo_outlies[:, 0], y=bidim_sentence_embeddings_wo_outlies[:, 1],
                    alpha=0.5, c=colors_wo_outliers[predictions_wo_outliers], s=1)
        plt.savefig("clustering_wo_outliers.png", bbox_inches='tight')
        # Scatter plot (with outliers) including sentence information (using plotly)
        x1_x2_sentence_cluster = np.concatenate([bidim_sentence_embeddings[:, 0].reshape(-1, 1).astype('float'),
                                                 bidim_sentence_embeddings[:, 1].reshape(-1, 1).astype('float'),
                                                 np.array(self.training_sentences).reshape(-1, 1),
                                                 predictions.reshape(-1, 1),
                                                 probs.reshape(-1, 1).astype('float')
                                                 ], axis=1)
        float_columns = ['x1', 'x2', 'probability']
        df = pd.DataFrame(x1_x2_sentence_cluster, columns=['x1', 'x2', 'sentence', 'cluster', 'probability'])
        for col_name in float_columns:
            df[col_name] = df[col_name].astype(float)
        # print(f"{df.head(5)}")
        # df.info()
        scatter_with_sentences(df)
        # Scatter plot (without outliers) including sentence information (using plotly)
        x1_x2_sentence_cluster_wo_outliers = np.concatenate(
            [bidim_sentence_embeddings_wo_outlies[:, 0].reshape(-1, 1).astype('float'),
             bidim_sentence_embeddings_wo_outlies[:, 1].reshape(-1, 1).astype('float'),
             np.array(self.training_sentences)[predictions != 0].reshape(-1, 1),
             predictions_wo_outliers.reshape(-1, 1),
             probs[predictions != 0].reshape(-1, 1).astype('float')
             ], axis=1)
        df = pd.DataFrame(x1_x2_sentence_cluster_wo_outliers, columns=['x1', 'x2', 'sentence', 'cluster', 'probability'])
        for col_name in float_columns:
            df[col_name] = df[col_name].astype(float)
        scatter_with_sentences(df, "plotly_sentences_wo_outliers.html")

    def plot_analysis(self):
        # self.hdbscan_model.single_linkage_tree_.plot()
        # plt.savefig("single_linkage_tree.png", bbox_inches='tight')
        self.hdbscan_model.condensed_tree_.plot(select_clusters=True,
                                                selection_palette=sns.color_palette('deep', 8))
        plt.savefig("condensed_tree.png", bbox_inches='tight')

    def plot_cluster_dist(self, cluster_preds):
        bars, height = np.unique(cluster_preds, return_counts=True)
        y_pos = np.arange(len(bars))
        # Create bars
        plt.bar(y_pos, height)
        # Create names on the x-axis
        plt.xticks(y_pos, bars, rotation=45, fontsize=6)
        plt.title("Sentences per cluster")
        plt.xlabel("Cluster ID (0: outliers)")
        plt.ylabel("N. sentences")
        plt.savefig("cluster_distribution.png", bbox_inches='tight')
        # Show graphic
        plt.show()
        # PLOT CLUSTER DISTRIBUTION WITHOUT OUTLIERS INFO
        plt.bar(y_pos[1:], height[1:])
        # Create names on the x-axis
        plt.xticks(y_pos[1:], bars[1:], rotation=90, fontsize=6)
        plt.title("Sentences per cluster (without outliers)")
        plt.xlabel("Cluster ID")
        plt.ylabel("N. sentences")
        plt.savefig("cluster_distribution_wo_outliers.png", bbox_inches='tight')
        # Show graphic
        plt.show()

    def save_hdbscan_model(self):
        filename = 'hdbscan_model.joblib'
        joblib.dump(self.hdbscan_model, filename)


def scatter_with_sentences(df, name=None):
    name = 'plotly_sentences.html' if name is None else name
    fig = px.scatter(df, x="x1", y="x2", hover_data=["sentence"], color='cluster', size='probability')
    fig.write_html(name)
    fig.show()


if __name__ == '__main__':
    df = get_clustering_results("..\\results\\HDBSCAN")
    breakpoint()
