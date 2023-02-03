from models import sbert_fine_tuning
from utils.utils import *


def main(args):
    fine_tuner = sbert_fine_tuning.SbertFineTuning(**vars(args))
    fine_tuner.read_silver_set()
    fine_tuner.load_bi_encoder_model()
    # Prepare evaluator (dev set for validation of hyperparameters; test set for final evaluation)
    fine_tuner.prepare_evaluator(dev=True)
    fine_tuner.fine_tune_sbert()


if __name__=='__main__':
    args = upload_args()
    main(args)