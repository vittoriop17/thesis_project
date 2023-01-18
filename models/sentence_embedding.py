from fse import Vectors, uSIF, SplitIndexedList, Average
from sentence_transformers import SentenceTransformer
import pickle
from data.data_loader import *
from models.pipeline import Pipeline

pretrained_bi_encoder_path = "D:\\UNIVERSITA\\KTH\\THESIS\\ProjectCode\\models\\ssp_cross_domain"


class AVGEmbedding:
    def __init__(self, train_sentences, word_embedding="paranmt-300", vecs=None, load_pretraines=None):
        self.vecs = Vectors.from_pretrained(word_embedding) if vecs is None else vecs
        self.model = Average(self.vecs)
        self.train_sentences = train_sentences
        if load_pretraines:
            raise NotImplementedError("'Load pretrained AVG embedding' not implemented")
        else:
            self._train_embeddings(train_sentences)

    def _train_embeddings(self, sentences):
        # list_word_in_sentences = [word_tokenize(sent) for sent in sentences_wo_punct]
        self.train_sentences = sentences
        self.model.train(SplitIndexedList(sentences))

    def embeddings(self, sentences):
        """
        Given a list of sentences_wo_punct, extract the list of sentence embeddings
        :param sentences: list of strings
        :return: list of sentence embeddings
        """
        return self.model.infer(SplitIndexedList(sentences))

    def most_similar_by_sentence_idx(self, sentence_idx : int):
        """
        :param sentence_idx: int. Index of the training sentence
        :return:
        """
        assert self.train_sentences, "Must train the model before checking the most similar sentences_wo_punct"
        assert 0 <= sentence_idx < len(self.train_sentences), f"Index out of bound. Sentence idx must be between 0 and {len(self.train_sentences)}"
        return self.model.sv.most_similar(sentence_idx, indexable=self.train_sentences)

    def most_similar_by_sentence(self, sentence):
        split_sentence = word_tokenize(sentence)
        return self.model.sv.similar_by_sentence(split_sentence, model=self.model, indexable=self.train_sentences)


    # def embeddings(self, sentences_wo_punct):
    #     return np.array([self.embedding(sentence) for sentence in sentences_wo_punct])
    #
    # def embedding(self, sentence):
    #     words_in_sentence = word_tokenize(sentence)
    #
    #     sentence_embedding = np.mean([self.vecs[word.lower()]
    #                                   if word.lower() in self.vecs else np.zeros(self.word_emb_len, )
    #                                   for word in words_in_sentence],
    #                                  axis=0)
    #     return sentence_embedding
    #

# We therefore considered embedding models that were trained using “all” or “paraphrase”
# training datasets and were designed as general-purpose models.
# see: https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e
class SentenceBertEmbedding:
    def __init__(self, bi_encoder_path="all-MiniLM-L12-v2"):  # N.B.: all-MiniLM-L12-v2 is a pre-trained model!!!!
        # Replace all-MiniLM-L12-v2 with the augmented sbert model when ready
        # Loading the augmented sbert model.
        # Check script 'train_sts_ssp_crossdomain.py' for details
        assert bi_encoder_path is not None, "bi_encoder_path not specified!"
        self.bi_encoder = SentenceTransformer(bi_encoder_path)
        print("Max Sequence Length:", self.bi_encoder.max_seq_length)

    def embeddings(self, sentences, save_to_file=None):
        # TODO - consider to pass the following arguments:
        # convert_to_numpy, normalize_embeddings
        embeddings = self.bi_encoder.encode(sentences, show_progress_bar=True)
        if save_to_file is not None:
            self._save_embeddings(sentences, embeddings, save_to_file)
        return embeddings

    def _save_embeddings(self, sentences, embeddings, file_path):
        with open(file_path, "wb") as fOut:
            pickle.dump({'sentences_wo_punct': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


class SifEmbedding:
    def __init__(self, train_sentences, word_embedding="paranmt-300", load_pretrained=None, vecs=None):
        self.vecs = Vectors.from_pretrained(word_embedding) if vecs is None else vecs
        self.model = uSIF(self.vecs)
        self.train_sentences = train_sentences
        if load_pretrained:
            raise NotImplementedError("'Load pretrained SIF embedding' not implemented")
        else:
            self._train_embeddings(train_sentences)

    def _train_embeddings(self, sentences):
        # list_word_in_sentences = [word_tokenize(sent) for sent in sentences_wo_punct]
        self.train_sentences = sentences
        self.model.train(SplitIndexedList(sentences))

    def embeddings(self, sentences):
        """
        Given a list of sentences_wo_punct, extract the list of sentence embeddings
        :param sentences: list of strings
        :return: list of sentence embeddings
        """
        return self.model.infer(SplitIndexedList(sentences))

    def most_similar_by_sentence_idx(self, sentence_idx : int, topn : int = 10):
        """
        :param topn : int or None, optional
            Number of top-N similar sentences_wo_punct to return, when `topn` is int. When `topn` is None,
            then similarities for all sentences_wo_punct are returned.
        :param sentence_idx: int. Index of the training sentence
        :return:
        """
        assert self.train_sentences, "Must train the model before checking the most similar sentences_wo_punct"
        assert 0 <= sentence_idx < len(self.train_sentences), f"Index out of bound. Sentence idx must be between 0 and {len(self.train_sentences)}"
        return self.model.sv.most_similar(sentence_idx, indexable=self.train_sentences, topn=topn)

    def most_similar_by_sentence(self, sentence):
        split_sentence = word_tokenize(sentence)
        return self.model.sv.similar_by_sentence(split_sentence, model=self.model, indexable=self.train_sentences)


if __name__ == '__main__':
    df_train, df_val, df_test = get_dataframes(PATH_DATA)
    test_pipe = Pipeline(df_test, [], split='test')
    sbert_emb_pretrained = SentenceBertEmbedding(pretrained_bi_encoder_path)
    test_pipe.df['sbert_emb_pretrained'] = sbert_emb_pretrained.embeddings(list(test_pipe.df.sentences_wo_punct))






