from models import sbert_fine_tuning
from utils.utils import *


def main(args):
    fine_tuner = sbert_fine_tuning.SbertFineTuning(**vars(args))
    fine_tuner.read_silver_set()
    fine_tuner.load_bi_encoder_model()
    # evaluate before fine_tuning
    print("Evaluation on test data before fine-tuning")
    fine_tuner.evaluate_sbert()
    fine_tuner.fine_tune_sbert()
    print("Evaluation on test data after fine-tuning")
    fine_tuner.evaluate_sbert()



if __name__=='__main__':
    args = upload_args()
    main(args)