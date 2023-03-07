from models import sbert_fine_tuning, clustering
from utils.utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch
import hdbscan
import umap


def main(args):
    if args.no_fine_tuning:
        clustering_pipe = clustering.ClusteringPipeline(**vars(args))
        if args.validate_hdbscan:
            clustering_pipe.evaluate()
            clustering_pipe.evaluate("precomputed")
        clustering_pipe.train_over_all_sentences()
        clustering_pipe.plot_clusters(clustering_pipe.original_sentence_embeddings)
    else:
        fine_tuner = sbert_fine_tuning.SbertFineTuning(**vars(args))
        fine_tuner.read_silver_set()
        fine_tuner.load_bi_encoder_model()
        # evaluate before fine_tuning
        print("Evaluation on test data before fine-tuning")
        fine_tuner.evaluate_sbert(load_finetuned=False)
        fine_tuner.fine_tune_sbert()
        print("Evaluation on test data after fine-tuning")
        fine_tuner.evaluate_sbert(load_finetuned=True)


if __name__=='__main__':
    args = upload_args()
    main(args)