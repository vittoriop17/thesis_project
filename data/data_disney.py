import spacy
import pandas as pd
import gzip
import json
from bs4 import BeautifulSoup
from html import unescape
from datetime import datetime
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from spacy.tokens import DocBin
import os
import inspect
from langdetect import detect
from bertopic import BERTopic
from silver_set_construction import *
from sklearn.model_selection import train_test_split


SPECIAL_SPACY_ENTITIES = ['ORDINAL', 'CARDINAL', 'MONEY', 'PERCENT', 'QUANTITY', 'TIME', 'DATE']
SPACY_DOCS_PATH = "..\\data\\{}_docs.spacy"
PATH_DATA = "DISNEY/en_reviews_disney.csv"
MIN_SENT_LEN = 3
MAX_SEN_LEN = 128
MAX_SENTENCES = 25_000

def extract_sentences_spacy(df, path_spacy_data=None, nlp=None):
    """
    Given a dataframe with the attribute 'reviewtext', applies the nlp_spacy pipeline to it.
    This is done only if path_spacy_data is None, or it points to a non-existing file. In this case,
    after applying the pipeline, it saves the results to the specified path (ready to be used next time)
    :param df: pandas dataframe. Must contain the attribute reviewText
    :param path_spacy_data: str. Path to the spacy data
    """
    if os.path.exists(path_spacy_data):
        nlp = spacy.blank("en")
        try:
            with open(path_spacy_data, "rb") as fin:
                bytes_data = fin.read()
                doc_bin = DocBin().from_bytes(bytes_data)
                df['doc'] = list(doc_bin.get_docs(nlp.vocab))
                df['spacy_sentences'] = [list(doc.sents) for doc in df.doc]
        except Exception as e:
            print(e)
            exit(-1)
    else:
        doc_bin = DocBin(attrs=["TAG", "POS", "LEMMA", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE"])
        if nlp is None:
            nlp = spacy.load("en_core_web_sm")
            nlp.add_pipe('sentencizer')
            nlp.add_pipe('merge_entities')
        # nlp.add_pipe('spacytextblob')
        assert hasattr(df, "reviewText")
        df['doc'] = [nlp(review) for review in tqdm(df.reviewText)]
        df['spacy_sentences'] = [list(doc.sents) for doc in df.doc]
        df['sentences_wo_punct'] = [str(sent) for sent in df.spacy_sentences]
        _ = [doc_bin.add(doc) for doc in df.doc]
        bytes_data = doc_bin.to_bytes()
        try:
            with open(path_spacy_data, "wb") as fout:
                fout.write(bytes_data)
        except Exception as e:
            print(e)
    # df['sentiment_polarity'] = list([doc._.polarity for doc in df.doc])
    df = df.explode('spacy_sentences')
    df['sentences_wo_punct'] = [str(sent) for sent in df.spacy_sentences]
    return nlp, df


def extract_bert_topics(df, nlp, topic_model=None):
    """
    Extract topics (sentence level). For this purpose, the BERTopic package will be used.
    If a pre-trained topic_model is passed, no fit will be applied
    :param nlp: SpaCY nlp
    :param topic_model: BERTopic model
    :return trained BERTopic model
    """
    topic_model = BERTopic(embedding_model=nlp, nr_topics="auto") if topic_model is None else topic_model
    topics, probs = topic_model.fit_transform(list(df.sentences_wo_punct))
    fig = topic_model.visualize_topics()
    fig.show()
    df['topic'] = list(topics)
    df['topic_probs'] = list(probs)
    return topic_model


# def replace_special_entities_and_lemmatize(df, lemmatize=True):
#     """
#     Given a dataframe, replace all the special spacy entities (see SPECIAL_SPACY_ENTITIES) with the
#     name of the entity itself. E.g.: It was worth the $30 I paid -> It was worth the money I paid
#     Plus, apply lemmatization. The two steps are integrated in one single function for convenience
#     :param df: pandas dataframe. Must contain the attribute 'doc'
#     """
#     assert hasattr(df, "doc"), "The dataframe does not contain the attribute 'doc'! Call function 'extract_sentences_spacy' first"
#     df['special_sentences'] = [replace_entities_in_sentence_and_lemmatize(sentence, lemmatize) for sentence in
#                                tqdm(df.spacy_sentences, inspect.stack()[0][3])]


# def replace_entities_in_sentence_and_lemmatize(sentence, lemmatize=True):
#     """
#     Replace special entities, lemmatize if required and remove punct
#     :param sentence: Spacy Span. Spacy sentence
#     :param lemmatize: bool.
#     :return: str. The preprocessed sentence
#     """
#     new_sentence = []
#     for token in sentence:
#         try:
#             if token.ent_type_ in SPECIAL_SPACY_ENTITIES:
#                 new_sentence.append(token.ent_type_.lower())
#             elif not token.is_punct:  # remove punct
#                 new_sentence.extend([token.lemma_.lower() if lemmatize else token.lower_])
#         except:
#             print(f"Exception occurent in {inspect.stack()[0][3]} for token {token}")
#             new_sentence.append(token.lower_)
#     return " ".join([t for t in new_sentence])


def extract_sentences_nltk(df):
    assert hasattr(df, "reviewText")
    df['sentences_wo_punct'] = [sent_tokenize(review) for review in tqdm(df.reviewText)]
    sentences = []
    for sents in df.sentences_wo_punct:
        sentences.extend(sents)


def extract_words(df):
    assert hasattr(df, "sentences_wo_punct"), "Attribute 'sentences_wo_punct' missing. Extract sentences_wo_punct before calling the function 'get_df_with_words'"
    df['words'] = [word_tokenize(sent) for sent in tqdm(df.sentences_wo_punct, inspect.stack()[0][3])]
    df['len_sentence'] = [len(words_in_sentence) for words_in_sentence in df.words]


def get_dataframe_from_gzip(gzip_filepath):
    data = []
    with gzip.open(gzip_filepath) as f:
        for l in f:
            data.append(json.loads(l.strip()))
    # convert list into pandas dataframe
    df = pd.DataFrame.from_dict(data)
    df = df.loc[~df.reviewText.isnull()]
    df.reset_index(inplace=True)
    return df


def filter_non_english_reviews(df, save_to_path=None):
    assert hasattr(df, "reviewText"),  f"Column 'reviewText' not found inside the dataframe. Set of columns: {df.columns}"
    print(f"Number of reviews before filtering out no-English reviews: {df.shape[0]}")
    df = df.iloc[[detect(reviewText)=='en' for reviewText in tqdm(df.reviewText, f"{inspect.stack()[0][3]}")]]
    print(f"Number of reviews after filtering out no-English reviews: {df.shape[0]}")
    if save_to_path:
        df.to_csv(save_to_path)


def get_dataframes(df_filepath, train_val_test=(0.7, 0.2, 0.1), n_total_sentences=None):
    """
    Load the dataframe from a csv file, then:
    1. clean the reviewText attribute (replace the special characters &...;, remove html tags, ...)
    2. sort the reviews based on the attribute unixReviewTime and add the attribute date to the dataframe
    3. split the dataset into train-val-test sets
    :param df_filepath: str. Path to the .csv file where the dataset is stored.
        For compatibility, it should at least contain the columns: reviewText, unixReviewTime (for splitting the set)
    :return:
        three dataframes: df_training, df_val, df_test
    """
    df = pd.read_csv(df_filepath)
    assert hasattr(df, "unixReviewTime"), f"Column 'unixReviewTime' not found inside the dataframe. Set of columns: {df.columns}"
    assert hasattr(df, "reviewText"),  f"Column 'reviewText' not found inside the dataframe. Set of columns: {df.columns}"

    # ------------------------- STEP 1 -----------------------------------
    # replace special characters (&...;) and html text
    df.reviewText = [BeautifulSoup(unescape(r), 'lxml').text for r in df.reviewText]

    # ------------------------- STEP 2 -----------------------------------
    # sort reviews by unix Time and add a new attribute to the dataframe: date
    df = df.sort_values(['unixReviewTime'], ascending=True).reset_index()
    df['date'] = [datetime.utcfromtimestamp(int(urt)).strftime('%Y-%m-%d %H:%M:%S') for urt in
                  df.unixReviewTime]

    # -------------------------- STEP 3 -----------------------------------
    # split data into training, validation and test sets
    train_samples = int(train_val_test[0] * df.shape[0])
    validation_samples = int(train_val_test[1] * df.shape[0])
    print(f"Number of \n"
          f"\ttrain samples: {train_samples}"
          f"\tvalidation samples: {validation_samples}"
          f"\ttest samples: {df.shape[0] - validation_samples - train_samples}")
    df_train = df.loc[:train_samples - 1]
    df_val = df.loc[train_samples: train_samples + validation_samples - 1]
    df_test = df.loc[train_samples+validation_samples:]

    return df_train, df_val, df_test


def save_sentences(sentences, filepath):
    with open(filepath, "w", encoding="utf-8") as fout:
        [fout.write(f"{sentence}\n") for sentence in sentences]


if __name__=='__main__':
    # df = get_dataframe_from_gzip("..\\data\\Software_5.json.gz")
    # sentences_wo_punct = get_sentences(df)
    en_dataset_path = "DISNEY\\en_reviews_disney.csv"
    path_spacy_data = "DISNEY\\all_docs.spacy"
    df, _, _ = get_dataframes(en_dataset_path, (1, 0, 0))
    nlp, df = extract_sentences_spacy(df, path_spacy_data)

    # remove duplicates
    print(f"Number of sentences (before removing duplicates): {len(df)}")
    df = df.drop_duplicates(subset='spacy_sentences')
    print(f"Number of unique sentences: {len(df)}")

    # FILTER SHORT AND LONG SENTENCES
    extract_words(df)
    n = df.shape[0]
    df = df[df.len_sentence > MIN_SENT_LEN]
    print(f"Number of sentences before filtering short sentences: {n}\tafter filtering: {df.shape[0]}")
    n = df.shape[0]
    df = df[df.len_sentence < MAX_SEN_LEN]
    print(f"Number of sentences before filtering long sentences: {n}\tafter filtering: {df.shape[0]}")

    # limit the number of sentences
    if not len(df) < MAX_SENTENCES:
        # keep the most recent sentences (df is sorted by date)
        df = df.iloc[-MAX_SENTENCES:]

    # split df into training-dev-test sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_dev = train_test_split(df_train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    print(f"Number of training sentences: {len(df_train)}")
    print(f"Number of dev sentences: {len(df_dev)}")
    print(f"Number of test sentences: {len(df_test)}")

    # Build training, dev, test sets with silver set construction object
    # N.b: also dev and test have automatically extracted labels
    # (used to check the quality of the fine-tuning... even though it would be better to have real labels)
    train_sents, dev_sents, test_sents = list(map(lambda x: str(x), df_train.spacy_sentences)), \
                                         list(map(lambda x: str(x), df_dev.spacy_sentences)), \
                                         list(map(lambda x: str(x), df_test.spacy_sentences)),

    SilverSetConstructor(train_sents, name='Disney dataset',
                         verbose=False, folder="DISNEY", task='regression')
    SilverSetConstructor(dev_sents, name='Disney dev dataset',
                         verbose=False, folder="DISNEY", task='regression', filepath="dev_set.tsv", n_pairs=1500)
    SilverSetConstructor(test_sents, name='Disney set dataset',
                         verbose=False, folder="DISNEY", task='regression', filepath="test_set.tsv", n_pairs=1500)
    # save sentences (n.b.: no sentence pairs!!)
    filepath = "DISNEY\\{}_sentences.txt"
    save_sentences(df_train.spacy_sentences, filepath.format("train"))
    save_sentences(df_dev.spacy_sentences, filepath.format("dev"))
    save_sentences(df_test.spacy_sentences, filepath.format("test"))
    # extract_bert_topics(df_train, nlp)
    # start_time = time.time()
    # sentences_wo_punct = extract_sentences_spacy(df_test)
    # print("--- Spacy solution: %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # sentences_wo_punct = extract_sentences_nltk(df_test)
    # print("--- NLTK solution: %s seconds ---" % (time.time() - start_time))
    # print()
