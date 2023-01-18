from utils.visualization import *
from models.sentence_embedding import *

# TODO - collapse all preprocessing in one single function!!
# the order of the steps should be the following:
# 1. extract sentences_wo_punct --> DONE
# 2. add attribute 'special_sentences':  --> DONE
#     a special sentence is a sentence with:
#       - replaced special entities (e.g.: all the dates will be modified into the work 'date')
#       - lemmatized words (if LEMMATIZE=True)
#       - no punctuation
#     special_sentences are used for sif_embeddings and glove-mean_embeddings
# 3. extract attributes 'word' (list of words in sentence) and 'len_sentence' (number of words in sentence)
# 4. filter sentences_wo_punct with less than MIN_SENT_LEN words

PATH_DATA = "data/DISNEY/en_reviews_disney.csv"

df_train, df_val, df_test = get_dataframes(PATH_DATA)

train_pipe = Pipeline(df_train)

umap_visualization(train_pipe.df, embedding_feature_name="sbert_embeddings_pretrained", n_components=2)

breakpoint()
#
# # ---- VISUALIZATION
# all_params = {
#     "n_neighbors": [150],
#     'embedding_feature_name': ["sbert_embeddings", "sif_embeddings", "avg_embeddings"],
#     "min_dist": [0.15],
#     "metric": ['cosine'],
#     "n_fit_points": [df_train.shape[0]]
# }
# grid = ParameterGrid(all_params)
# for params in tqdm(grid, "UMAP visualization"):
#     title = f"  ------- {params['embedding_feature_name']} ------- \n" \
#             f"UMAP training sentences_wo_punct: {params['n_fit_points']}.\tTotal number of sentences_wo_punct: {df_train.shape[0]}\n" \
#             f"n_neighbors: {params['n_neighbors']}.\n" \
#             f"min_dist: {params['min_dist']}.\n"
#     umap_visualization(df_train, n_components=2, title=title, **params)
