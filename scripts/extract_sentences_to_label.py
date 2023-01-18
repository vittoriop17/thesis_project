from models.sentence_embedding import *
from data.data_loader import *
import numpy as np
import csv


PATH_DEST = "..\\data\\{}_sentences_to_label.csv"
PATH_DATA = "../data/DISNEY/en_reviews_disney.csv"
SEED = 1234567890
N_SENTENCES = (500, 200, 100)


np.random.seed(SEED)
dfs = get_dataframes(PATH_DATA, (0.7, 0.2, 0.1))
train_pipe = Pipeline(dfs[0], embeddings=[])
val_pipe = Pipeline(dfs[1], embeddings=[], split='val', vecs=train_pipe.vecs, nlp=train_pipe.nlp)
test_pipe = Pipeline(dfs[2], embeddings=[], split='val', vecs=train_pipe.vecs, nlp=train_pipe.nlp)

for split, n_sentences in zip(['train', 'val', 'test'], N_SENTENCES):
    df = train_pipe.df if split is 'train' else val_pipe.df if split is 'val' else test_pipe.df
    sentence_idxs = np.random.choice(df.shape[0], size=n_sentences, replace=False)
    with open(PATH_DEST.format(split), 'w', newline='', encoding='utf-8') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['sentence_idx', 'sentence'])
        for s_idx in sentence_idxs:
            csv_out.writerow([s_idx, df.sentences_wo_punct.iloc[s_idx]])