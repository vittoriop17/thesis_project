from data.data_loader import extract_sentences_spacy, SPACY_DOCS_PATH, replace_special_entities_and_lemmatize, \
    extract_words, MIN_SENT_LEN, MAX_SEN_LEN
from models import sentence_embedding as se


class Pipeline:
    def __init__(self, df, embeddings=None, split="train", nlp=None, vecs=None):
        # self.df_train, self.df_val, self.df_test = get_dataframes(reviews_filepath)
        if embeddings is None:
            embeddings = ['usif', 'avg', 'sbert', 'sbert_pretrained']
        self.df = df.copy()
        # TODO - remove next line. Save dataframe with sentences_wo_punct already split!!!!!!!
        self.nlp, self.df = extract_sentences_spacy(self.df, SPACY_DOCS_PATH.format(split), nlp)

        # TODO - remove next line
        # self.df = self.df.iloc[:10_000]

        # the next instruction adds the attribute 'special_sentences' to the dataframe
        # replace_special_entities_and_lemmatize(self.df, lemmatize=False)
        extract_words(self.df)
        # FILTER SHORT AND LONG SENTENCES
        n = self.df.shape[0]
        self.df = self.df[self.df.len_sentence >= MIN_SENT_LEN]
        print(f"Number of sentences before filtering short sentences: {n}\tafter filtering: {self.df.shape[0]}")
        n = self.df.shape[0]
        self.df = self.df[self.df.len_sentence <= MAX_SEN_LEN]
        print(f"Number of sentences before filtering long sentences: {n}\tafter filtering: {self.df.shape[0]}")

        self.vecs = vecs
        self.embeddings = embeddings
        # ---- EMBEDDINGS
        self.sif_em, self.avg_emb, self.sbert_emb = None, None, None
        # SIF
        if 'usif' in embeddings:
            self.sif_emb = se.SifEmbedding(list(self.df.sentences_wo_punct), vecs=self.vecs)
            self.df['sif_embeddings'] = list(self.sif_emb.embeddings(list(self.df.sentences_wo_punct)))
            self.vecs = self.sif_emb.vecs if self.vecs is None else self.vecs
        # AVG
        if 'avg' in embeddings:
            self.avg_emb = se.AVGEmbedding(list(self.df.sentences_wo_punct), vecs=self.vecs)
            self.df['avg_embeddings'] = list(self.avg_emb.embeddings(list(self.df.sentences_wo_punct)))
            self.vecs = self.avg_emb.vecs if self.vecs is None else self.vecs
        # SBERT
        if 'sbert' in embeddings:
            self.sbert_emb = se.SentenceBertEmbedding()
            self.df['sbert_embeddings'] = list(self.sbert_emb.embeddings(list(self.df.sentences_wo_punct)))
        if 'sbert_pretrained' in embeddings:
            self.sbert_emb_pretrained = se.SentenceBertEmbedding(se.pretrained_bi_encoder_path)
            self.df['sbert_embeddings_pretrained'] = list(self.sbert_emb.embeddings(list(self.df.sentences_wo_punct)))

    def get_sentence(self, idx, special=False):
        return self.df.sentences_wo_punct.iloc[idx] if not special else self.df.special_sentences.iloc[idx]
