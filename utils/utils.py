import re

import pandas as pd
import spacy
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import seaborn as sns
import os

# nlp = spacy.load("en_core_web_md")
#
# tag_re = re.compile(r"<(.).*\/(\1)>")
# text = "I have had Dreamweaver MX2004 since it came out back then. Spent years with it. " \
#        "Feel like I know it real well, but I am still familiar with tables as opposed to CSS. " \
#        "So I thought this would be a great introduction, and it is. The problem is that while I am looking at the video," \
#        " and that simplifies things a lot, I am not getting the intuitive explanation as to why things work the way they do." \
#        " I understand just knowing how to work them is sufficient. " \
#        "I'm tempted to delve into the rich full attributes of Dreamweaver CS5 as explained by this video, " \
#        "albeit difficult to understand the more advanced features, but I won't because this is about the video." \
#        "\n\nThe opening salvo is chock full of... for example, this is the URL; this is where you type in the address bar etc. " \
#        "Only when you start to get into areas such as CSS do you get an introduction that for me is beneficial, " \
#        "and that is only because I am somewhat of a newbie to it. So there you have it, " \
#        "do you get an advanced lesson that flies through things and most find it hard to understand, " \
#        "or do you get this one and appreciate the fact that she walks you through everything step by step, " \
#        "even at the risk of boring the more in the know people, but finally arriving at the section you need to learn," \
#        " and then appreciate that she is not moving at lightning speed.\n\nI'm holding off buying Dreamweaver CS5, " \
#        "although I got the upgrade to Photoshop CS5 and love it, because my MX2004 is not an eligible upgrade version." \
#        " If I could qualify for the upgrade I would grab it, but the learning curve of CSS, " \
#        "and the steep price for the full version when I have MX2004 already is giving me some hesitation. " \
#        "If I was to upgrade, or should I say take the leap to the full version, this tutorial is the one I would use, " \
#        "and I also recommend the&nbsp;<a data-hook=\"product-link-linked\" class=\"a-link-normal\" href=\"/Adobe-Dreamweaver-CS5-on-Demand/dp/0789744449/ref=cm_cr_arp_d_rvw_txt?ie=UTF8\">Adobe Dreamweaver CS5 on Demand</a>&nbsp;which makes a very good reference guide.\nAt some point I will get the new version of CS5 Dreamweaver, and at that point I will do the entire video series, and have a much more in depth analysis of it. But from what I can see after viewing a bunch of the video tutorials, it is the easiest way to indoctrinate you to the newest version. So I recommend it."
# text = re.sub(tag_re, "<TAG>", text)
# text = re.sub("&nbsp;", " ", text)
#
# nlp.tokenizer.add_special_case("<TAG>", [{ORTH: "<TAG>"}])
# doc = nlp(text)
# for token in doc:
#     print(token)
#
# for sent in doc.sents:
#     print(sent)
#
# for sent in doc.sents:
#     tokens_wo_stopwords = [token for token in sent if not token.is_stop]
# print(f"Doc mean vector: {doc.vector}")
#
# # Merge multi-word tokens (e.g.: New York, United Kingdom, ...)
# # This only matches multi-words that are entities
# with doc.retokenize() as retokenizer:
#     for ent in doc.ents:
#        retokenizer.merge(doc[ent.start:ent.end], attrs={"LEMMA": ent.text})

from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# sentences_wo_punct = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']
# sentence_embeddings = model.encode(sentences_wo_punct)
# for sentence, embedding in zip(sentences_wo_punct, sentence_embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")
import argparse
import json


def sentence_len_distribution(df):
    assert hasattr(df, "len_sentence"), "Attribute 'len_sentence' missing. Call function 'extract_words' before"
    df.len_sentence.value_counts()
    df.len_sentence.value_counts().plot.bar()
    plt.xticks(fontsize=6, rotation=90)
    plt.show()


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2)) if norm(v1) != 0 and norm(v2) != 0 else 0


def euclidean_similarity(v1, v2):
    return np.linalg.norm(v1 - v2)


def countplot_sentence_scores(folder_path, train_filename, title):
    """
    Countplot where:
    score 0: unrelated sentences;
    score 1: related sentence.
    In the same plot, there are the aggregated data coming from 3 different sources:
    -training set;
    -dev set;
    -test set.
    """
    train_path, dev_path, test_path = os.path.join(folder_path, train_filename), \
                                      os.path.join(folder_path, "dev_set.tsv"), \
                                      os.path.join(folder_path, "test_set.tsv")
    fig, ax = plt.subplots(1, 3)
    df_train = pd.read_csv(train_path, sep='\t')
    df_dev = pd.read_csv(dev_path, sep='\t')
    df_test = pd.read_csv(test_path, sep='\t')
    sns.set_style('darkgrid')
    sns.countplot(x='score', data=binarize_scores(df_train), ax=ax[0]).set_title("Training set")
    sns.countplot(x='score', data=binarize_scores(df_dev), ax=ax[1]).set_title("Dev set")
    sns.countplot(x='score', data=binarize_scores(df_test), ax=ax[2]).set_title("Test set")
    plt.show()


def binarize_scores(df):
    df.score = [1 if score > 0.5 else 0 for score in df.score]
    return df


def upload_args(file_path=None):
    parser = argparse.ArgumentParser(description=f'Arguments for sbert fine-tuning or clustering',
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument("--name", required=False, type=str, help="Name of the experiment")
    # ----- SBERT FINE-TUNING PARAMETERS
    parser.add_argument("--num_epochs", required=False, type=int,
                        help="Number of epochs")
    parser.add_argument("--scenario", required=False, type=int,
                        choices=[1, 2],
                        help="Fine tuning scenario. Valid options: 1, 2")
    parser.add_argument("--loss_type", required=False, type=str,
                        choices=['softmax', 'cosine', 'contrastive', 'multiple_neg_ranking'],
                        help="Loss type used for the sentence-bert fine-tuning. "
                             "Valid options: softmax, cosine, multiple_neg_ranking, contrastive")
    parser.add_argument("--evaluator_type", required=False, type=str,
                        choices=['binary', 'regression', 'multilabel_accuracy'],
                        help="Evaluator type used for the evaluation of the sentence-bert model. "
                             "Valid options: binary, regression, multilabel_accuracy")
    parser.add_argument("--batch_size", required=False, type=int,
                        help="Batch size used for sentence-bert fine-tuning.")
    parser.add_argument("--max_seq_length", default=128, required=False, type=int,
                        help="Max sequence length for sentence-bert input sentences. LOnger sentences will be truncated")
    parser.add_argument("--base_model", required=False, type=str,
                        help="Base sentence-bert model. See HuggingFace or sentence-transformers doc for available options")
    parser.add_argument("--task", required=False, type=str,
                        choices=['classification', 'regression'],
                        help="Task type used for sentence-bert fine tuning. Valid options: 'classification', 'regression'")

    # ----- DATASET INFO (name, paths, ...)
    parser.add_argument("--silver_set_path", required=False, type=str,
                        help="Train dataset path (with silver labels obtained with SIF model).")
    parser.add_argument("--dataset_name", required=False, type=str,
                        choices=['STS', 'MRPC', 'DISNEY'],
                        help="Name of the dataset used for experiments. Valid options: STS, MRPC, DISNEY")
    parser.add_argument("--dev_set_path", required=False, type=str,
                        help="Validation dataset path (with real labels).")
    parser.add_argument("--test_set_path", required=False, type=str,
                        help="Test dataset path (with real labels).")

    # ----- KBinsDiscretizer params (for STS and DISNEY datasets)
    parser.add_argument("--strategy", required=False, type=str,
                        choices=['uniform', 'quantile'],
                        help="Strategy adopted for the KBinsDiscretizer (check sklearn doc). "
                             "Accepted values: 'uniform' and 'quantile'. 'kmeans' not accepted.")
    parser.add_argument("--n_bins", required=False, type=int,
                        help="Number of bins used for score discretization (for KBinsDIscretizer)")

    # ----- Additional parameters
    parser.add_argument("--evaluation_only", action='store_true', required=False,
                        help="Specify if only test is required")
    parser.add_argument("--bi_encoder_path", required=False, type=str,
                        help="Path to pre-trained bi-encoder. Used when evaluation_only is set to true")
    parser.add_argument("--config_file_path", required=False, type=str,
                        help="Path to the configuration file with arguments. "
                             "It is an alternative to the command line arguments."
                             "Be careful, the command line arguments"
                             " have priority over the arguments indicated in the configuration file")
    parser.add_argument("--no_fine_tuning", action="store_true", default=False,
                        help="Boolean flag. If specified, s-bert fine-tuning is not applied")

    # ------- Arguments for clustering (DISNEY DATASET ONLY)
    parser.add_argument("--train_sentences_path", type=str,
                        help="Path to the training sentences (used to train the clustering model)")
    parser.add_argument("--validate_umap", action="store_true", default=False,
                        help="Boolean flag. If specified, apply UMAP model validation "
                             "(min_dist and n_components are fine-tuned, based on trustworthiness metric)")
    parser.add_argument("--validate_hdbscan", action="store_true", default=False,
                        help="Boolean flag. If specified, apply HDBSCAN model validation with DBCV metric")
    parser.add_argument("--n_components", default=5, type=int,
                        help="number of components for UMAP dimensionality reduction")
    parser.add_argument("--umap_min_dist", type=float,
                        help="Min_dist parameter for umap")
    parser.add_argument("--umap_n_neighbors", type=int,
                        help="number of neighbors for UMAP dimensionality reduction")
    parser.add_argument("--umap_metric", default='cosine', type=str,
                        help="Distance metric used for UMAP dimensionality reduction")
    parser.add_argument("--hdbscan_min_samples", type=int,
                        help="min_samples argument for HDBSCAN")
    parser.add_argument("--hdbscan_min_cluster_size", type=int,
                        help="min_cluster_size argument for HDBSCAN")
    parser.add_argument("--hdbscan_metric", type=str,
                        help="Distance metric for HDBSCAN")
    parser.add_argument("--hdbscan_cluster_method", type=str,
                        help="Cluster method selection for HDBSCAN")
    parser.add_argument("--hdbscan_epsilon", type=float,
                        help="Cluster method epsilon for HDBSCAN")
    args = parser.parse_args()
    file_path = getattr(args, 'config_file_path') if getattr(args, 'config_file_path') is not None else file_path
    args = upload_args_from_json(args, file_path) if file_path is not None else args
    print(args)
    return args


def upload_args_from_json(args, file_path="config.json"):
    if args is None:
        parser = argparse.ArgumentParser(description=f'Arguments from json')
        args = parser.parse_args()
    json_params = json.loads(open(file_path).read())
    for option, option_value in json_params.items():
        # do not override pre-existing arguments, if present.
        # In other terms, the arguments passed through CLI have the priority
        if hasattr(args, option) and getattr(args, option) is not None:
            continue
        if option_value == 'None':
            option_value = None
        if option_value == "True":
            option_value = True
        if option_value == "False":
            option_value = False
        setattr(args, option, option_value)
    return args


def save_args_to_json(dictionary, filepath):
    with open(filepath, 'w') as f:
        json.dump(dictionary, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    countplot_sentence_scores("..\\data\\STS", "silver_set_regression_cosine.tsv", None)
    countplot_sentence_scores("..\\data\\MRPC", "silver_set_classification0.5_cosine.tsv", None)
    breakpoint()
