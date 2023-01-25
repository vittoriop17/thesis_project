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
    return np.dot(v1, v2) / (norm(v1) * norm(v2)) if norm(v1)!=0 and norm(v2)!=0 else 0


def euclidean_similarity(v1, v2):
    return np.linalg.norm(v1-v2)


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


def upload_args(file_path="config.json"):
    parser = argparse.ArgumentParser(description=f'Arguments from json')
    parser.add_argument("--name", required=False, type=str, help="Name of the experiment "
                                                                "(e.g.: 'evaluate preprocessing: recenter_wrt_frame' or"
                                                                " 'test sequence length: 300')")
    parser.add_argument("--n_epochs", required=False, type=int, help="Number of epochs")
    parser.add_argument("--test_only", required=False, type=bool, help="Specify if only test is required")
    parser.add_argument("--save_model", required=False, type=bool, default=False, help="Boolean flag: set it if you want to save the model")
    parser.add_argument("--input_size", required=False, type=int, help="Input size of a singular time sample")
    parser.add_argument("--hidden_size", required=False, type=int)
    parser.add_argument("--train_only", required=False, type=bool, help="If True, apply only train. The Test score is evaluated at the end of the training. Otherwise, apply train and evaluation")
    parser.add_argument("--num_layers", required=False, type=int)
    parser.add_argument("--sequence_length", required=False, type=int)
    parser.add_argument("--lr", required=False, type=float)
    parser.add_argument("--batch_size", required=False, type=int)
    parser.add_argument("--train", required=False, type=bool)
    parser.add_argument("--video", required=False, type=str, help="Video path. Video used for evaluation of results")
    parser.add_argument("--multitask", required=False, type=bool, help="Training the multitask network or the classificatio network")
    parser.add_argument("--train_dataset_path", required=False, type=str, help="Train dataset path.")
    parser.add_argument("--checkpoint_path", required=False, type=str, help="path to checkpoint")
    parser.add_argument("--load_model", required=False, type=bool, help="Specify if load an existing model or not. If 'True', checkpoint_path must be specified as well")
    parser.add_argument("--test_dataset_path", required=False, type=str, help="Test dataset path.")
    parser.add_argument("--preprocess", required=False, type=str, help="Possible options: "
                                                                       "recenter: apply centering by frame and normalization by coordinate "
                                                                       "normalize: apply only normalization by coordinate "
                                                                       "recenter_by_sequence: apply centering by frame considering the mean-center of the current sequence "
                                                                       "... otherwise, do nothing (raw trajectories)")
    parser.add_argument("--dropout", required=False, type=float, help="Network dropout.")
    parser.add_argument("--alpha", required=False, type=float, help="Parameter for weighting the 2 losses (needed for training)")
    parser.add_argument("--stride", required=False, type=float, help="Window stride (for sequence definition)."
                                                                     "To be intended in relative terms (perc %).")
    parser.add_argument("--with_conv", required=False, type=bool, help="Specify if use 1-D Convolution, in order"
                                                                      " to preprocess the input sequences")
    args = parser.parse_args()
    args = upload_args_from_json(args, file_path)
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


