from models.sentence_embedding import *
from data.data_loader import *
import csv
from utils.utils import *
from sentence_transformers.cross_encoder import CrossEncoder


# TODO : add score computation (with SIF and with cross-encoder)


PATH_DEST = "..\\data\\{}_review_sentence_pairs_w_{}_score.csv"
PATH_DATA = "../data/DISNEY/en_reviews_disney.csv"
SEED = 1234567890
N_RANDOM_SENTENCES = 1_000
N_PAIRS_PER_SENTENCE = 5
SIF_PAIRS = True  # if true, for each random sentence,
                  # N_PAIRS_PER_SENTENCE sentences_wo_punct are sampled from the list of 'most_similar_sentences'.
                  # otherwise, N_PAIRS_PER_SENTENCE sentences_wo_punct are sampled from the list of all sentences_wo_punct
SIF_SCORE = False  # if true, assign a BINARY value to each sentence-pair, using the SIF similarity score
                  # if False, assign this value using the pre-trained CrossEncoder
cross_encoder_path = "D:\\UNIVERSITA\\KTH\\THESIS\\ProjectCode\\models\\stsb_cross_encoder__bert_base_uncased"
pairs = 'SIF' if SIF_PAIRS else 'RANDOM'
score = 'SIF' if SIF_SCORE else 'CROSSENCODER'
np.random.seed(SEED)

df_train, df_val, df_test = get_dataframes(PATH_DATA)
train_pipe = Pipeline(df_train, embeddings=['usif'])
sentence_pairs = []
sentence_pairs_w_score = []  # list of tuples. (sentence1, sentence2, score)
sentence_idxs = np.random.choice(train_pipe.df.shape[0], size=N_RANDOM_SENTENCES, replace=False)

for idx in sentence_idxs:
    s1 = train_pipe.get_sentence(idx)
    if SIF_PAIRS:
        most_similar = train_pipe.sif_emb.most_similar_by_sentence_idx(idx, N_PAIRS_PER_SENTENCE)
        # s2_idxs = np.random.choice(len(most_similar), size=N_PAIRS_PER_SENTENCE, replace=False)
        if SIF_SCORE:
            sentence_pairs_w_score.extend([(s1, train_pipe.get_sentence(most_similar[s2_idx][1]), 0 if most_similar[s2_idx][2] < 0.5 else 1) for s2_idx in range(N_PAIRS_PER_SENTENCE)])
        else:
            sentence_pairs.extend([(s1, train_pipe.get_sentence(most_similar[s2_idx][1])) for s2_idx in range(N_PAIRS_PER_SENTENCE)])
    else:
        s2_idxs = np.random.choice(train_pipe.df.shape[0], size=N_PAIRS_PER_SENTENCE, replace=False)
        s1_emb = train_pipe.sif_emb.embeddings([s1])[0]
        s2_embs = train_pipe.sif_emb.embeddings([train_pipe.get_sentence(s2_idx) for s2_idx in s2_idxs])
        if SIF_SCORE:
            sentence_pairs_w_score.extend([(s1, train_pipe.get_sentence(s2_idx), 0 if cosine_similarity(s1_emb, s2_emb) < 0.5 else 1) for s2_idx, s2_emb in zip(s2_idxs, s2_embs)])
        else:
            sentence_pairs.extend([(s1, train_pipe.get_sentence(s2_idx)) for s2_idx, s2_emb in zip(s2_idxs, s2_embs)])

if not SIF_SCORE:
    cross_encoder = CrossEncoder(cross_encoder_path)
    silver_scores = cross_encoder.predict(sentence_pairs)
    # All model predictions should be between [0,1]
    assert all(0.0 <= score <= 1.0 for score in silver_scores)
    binary_silver_scores = [1 if score >= 0.5 else 0 for score in silver_scores]
    sentence_pairs_w_score.extend([(s_pair[0], s_pair[1], score) for s_pair, score in zip(sentence_pairs, binary_silver_scores)])


def remove_duplicates(x):
    return list(dict.fromkeys(x))


sentence_pairs_w_score = remove_duplicates(sentence_pairs_w_score)

with open(PATH_DEST.format(pairs, score), 'w', newline='', encoding='utf-8') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['sentence1', 'sentence2', 'score'])
    for row in sentence_pairs_w_score:
        csv_out.writerow(row)
