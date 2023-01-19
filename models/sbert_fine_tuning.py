from torch.utils.data import DataLoader
from sentence_transformers import models, losses, util, LoggingHandler, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
from zipfile import ZipFile
import logging
import csv
import sys
import torch
import math
import gzip
import os
import pandas as pd

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


class SbertFineTuning:
    def __init__(self, silver_set_path, dev_set_path, test_set_path, dataset_name, scenario,
                 evaluation_only=False, bi_encoder_path=None, base_model=None):
        os.chdir("models")
        logging.info("Start checking argument...")
        self.sanity_check(dataset_name, scenario, bi_encoder_path, evaluation_only, silver_set_path)
        logging.info("...check done")
        self.scenario = scenario
        self.dataset_name = dataset_name
        self.dev_set_path = dev_set_path
        self.test_set_path = test_set_path
        self.base_model = 'bert-base-uncased' if base_model is None else base_model
        # params taken from AugSBERT paper
        self.batch_size = 16
        self.num_epochs = 1
        self.max_seq_length = 128
        self.use_cuda = torch.cuda.is_available()
        self.silver_set_path = silver_set_path
        # used to save the fine-tuned model
        self.bi_encoder_path = \
            os.path.join(dataset_name, f"scenario_{scenario}",
                         "bi_encoder" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) \
                if evaluation_only == False else bi_encoder_path
        # TODO - save model configuration (hyperparameters, ...). Remember to create folders (os.mkdirs)
        # Load bi-encoder model
        self.bi_encoder_model = None
        if evaluation_only:
            self.bi_encoder_model = SentenceTransformer(self.bi_encoder_path)
            self.prepare_evaluator(dev=False)
            self.evaluate_sbert()
        else:
            # Read Silver Data
            # TODO - training + dev set
            self.silver_data = []
            self.silver_scores = []
            self.binary_silver_scores = []
            self.read_silver_set()
            # Load bi-encoder
            self.load_bi_encoder_model()
            # Prepare evaluator (dev set for validation of hyperparameters; test set for final evaluation)
            # TODO - now it only loads dev data. Modify the code in order to load test data also
            #  remember that you should also modify the read_silver_set as well.
            #  It should contain training_silver_set + dev_silver_set (dev_silver_set must be generated first!)
            self.evaluator = None
            self.prepare_evaluator(dev=True)
            self.fine_tune_sbert()
            # Evaluate on test set
            self.prepare_evaluator(dev=False)
            self.bi_encoder_model.evaluate(self.evaluator)
        os.chdir("..")

    def sanity_check(self, dataset_name, scenario, bi_encoder_path, evaluation_only, silver_set_path):
        assert str.upper(dataset_name) in ['MRPC', 'STS', 'DISNEY'], \
            f"Invalid value for argument 'dataset_name'!, Expected one of these: MRPC, STS, DISNEY. Found {dataset_name}"
        assert scenario in [1, 2], f"Invalid value for argument 'scenario'. Expected 1 or 2. Found: {scenario}"
        assert bi_encoder_path is not None if evaluation_only else True, \
            "Argument 'bi_encoder_path' required if 'evaluation_only==True'!"
        assert os.path.exists(bi_encoder_path) if evaluation_only else True, \
            f"Pre-trained bi-encoder model not found! Provided path: {bi_encoder_path}"
        assert os.path.exists(silver_set_path), f"Target dataset not found. Given path: {silver_set_path}"

    def load_bi_encoder_model(self):
        logging.info("Loading bi-encoder model: {}".format(self.base_model))
        # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
        word_embedding_model = models.Transformer(self.base_model, max_seq_length=self.max_seq_length)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        self.bi_encoder_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def read_silver_set(self):
        logging.info("Loading Silver Set (for scenario {})".format(self.scenario))
        if self.scenario == 2:
            with open(self.silver_set_path, encoding='utf8') as fin:
                reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
                for row in reader:
                    self.silver_data.append([row['sentence_1'], row['sentence_2']])
                    score = float(row['score'])
                    assert 0 <= score <= 1, f"Found invalid score: {score}"
                    self.silver_scores.append(score)
                    self.binary_silver_scores.append(1 if score >= 0.5 else 0)
        else:
            # TODO - implement dataset preparation for scenario 1
            print("Read silver set for scenario 2 not implemented!")
            exit(-10)

    def prepare_evaluator(self, dev=True):
        logging.info(f"Preparing evaluator (for model validation). Data from {'dev set' if dev else 'test set'}")
        sentences1 = []
        sentences2 = []
        labels = []
        eval_data_path = self.dev_set_path if dev else self.test_set_path
        with open(eval_data_path, encoding='utf8') as fin:
            reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                sentences1.append(row['sentence_1'])
                sentences2.append(row['sentence_2'])
                labels.append(int(float(row['score'])))  # Note: binary label
        self.evaluator = BinaryClassificationEvaluator(sentences1, sentences2, labels)

    def fine_tune_sbert(self):
        """
        Apply sbert fine-tuning, using the pre-loaded silver set
        :return:
        """
        logging.info(f"Fine tune bi-encoder: over labeled Silver Set ({self.dataset_name}), using binary scores")
        # Convert the dataset to a DataLoader ready for training
        train_data = \
            list(InputExample(texts=[data[0], data[1]], label=score)
                 for (data, score) in zip(self.silver_data, self.binary_silver_scores))
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.bi_encoder_model)

        # Configure the training.
        # warmup_steps - training configuration taken from augSBERT
        warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Train the bi-encoder model
        self.bi_encoder_model.fit(train_objectives=[(train_dataloader, train_loss)],
                                  evaluator=self.evaluator,
                                  epochs=self.num_epochs,
                                  evaluation_steps=1000,
                                  warmup_steps=warmup_steps,
                                  output_path=self.bi_encoder_path
                                  )

    def evaluate_sbert(self):
        logging.info("Starting evaluation on test data...\n")
        self.bi_encoder_model.evaluate(self.evaluator)


if __name__ == '__main__':
    silver_set_path = "../data/STS/silver_set_regression_cosine.tsv"
    dev_set_path = "../data/STS/dev_set.tsv"
    test_set_path = "../data/STS/test_set.tsv"

    fine_tuner = SbertFineTuning(silver_set_path=silver_set_path,
                                 dev_set_path=dev_set_path,
                                 test_set_path=test_set_path,
                                 dataset_name="STS",
                                 scenario=2)
