from torch.utils.data import DataLoader
from sentence_transformers import models, losses, util, LoggingHandler, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CEBinaryClassificationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
import logging
import csv
import sys
import os
import math
import numpy as np
import argparse


class CrossEncoderFineTuner:
    def __init__(self, silver_set_path, dev_set_path, dataset_name, task,
                 base_model='bert-base-uncased', batch_size: int = 16, num_epochs: int = 10, max_seq_length: int = 128,
                 num_samples=1000, only_labeling=False, cross_encoder_path=None):
        """

        :param silver_set_path:
        :param dev_set_path:
        :param dataset_name:
        :param base_model:
        :param num_epochs:
        :param task: str. {"regression", "classification"}
        :param num_samples: int. Number of training samples to use for fine-tuning
        """
        os.chdir("models") if os.getcwd().endswith('ProjectCode') or os.getcwd().endswith('thesis_project') else None
        assert cross_encoder_path is not None if only_labeling else True, f"Must provide the path to an existing cross encoder model when 'only_labeling' is set to True"
        self.cross_encoder_path = os.path.join(dataset_name, dataset_name + "_crossencoder_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) \
            if not only_labeling else cross_encoder_path

        self.silver_data, self.all_silver_data = None, None
        self.binary_silver_scores = None
        self.cross_encoder_model = None
        self.silver_scores = None
        self.dataset_name = dataset_name
        self.silver_set_path = silver_set_path
        self.dev_set_path = dev_set_path
        self.model_base = base_model
        self.num_epochs = num_epochs
        self.task = task
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.only_labeling = only_labeling
        self.read_data()
        self.load_model(from_file=True if only_labeling else False)

    def load_model(self, from_file=False):
        self.cross_encoder_model = CrossEncoder(self.model_base, num_labels=1) if not from_file \
            else CrossEncoder(self.cross_encoder_path)

    def read_data(self):
        all_sentence_pairs = []
        all_scores, all_binary_scores = [], []
        with open(self.silver_set_path, encoding='utf8') as fin:
            reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                all_sentence_pairs.append([row['sentence_1'], row['sentence_2']])
                score = float(row['score'])
                assert 0 <= score <= 1, f"Found invalid score: {score}"
                all_scores.append(score)
                all_binary_scores.append(1 if score >= 0.5 else 0)
        assert len(
            all_sentence_pairs) >= self.num_samples, f"Number of training samples too large. Max {len(all_sentence_pairs)} samples"
        training_idxs = np.random.choice(len(all_sentence_pairs), size=self.num_samples, replace=False)
        self.silver_data = np.array(all_sentence_pairs)[training_idxs]
        self.silver_scores = np.array(all_scores)[training_idxs]
        self.binary_silver_scores = np.array(all_binary_scores)[training_idxs]
        # save also the other sentence pairs (those not selected). They will be used inside 'label_set()' function
        self.all_silver_data = all_sentence_pairs

    def prepare_evaluator(self):
        logging.info(f"Preparing evaluator")
        dev_samples = []
        eval_data_path = self.dev_set_path
        with open(eval_data_path, encoding='utf8') as fin:
            reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if self.task == 'regression':
                    score = (float(row['score']))
                elif self.task == 'classification':
                    score = (0 if float(row['score'])<0.5 else 1)  # Note: binary label
                dev_samples.append(InputExample(texts=[row['sentence_1'], row['sentence_2']], label=score))
        if self.task == 'regression':
            self.evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='dev')
        elif self.task == 'classification':
            self.evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='dev')

    def fine_tune(self):
        train_samples = []
        scores = self.silver_scores if self.task == 'regression' else self.binary_silver_scores
        train_samples.extend(InputExample(texts=[row[0], row[1]], label=score) for row, score in
                              zip(self.silver_data, scores))
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=self.batch_size)
        self.prepare_evaluator()
        warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1)  # 10% of train data for warm-up
        self.cross_encoder_model.fit(train_dataloader=train_dataloader,
                                     evaluator=self.evaluator,
                                     epochs=self.num_epochs,
                                     evaluation_steps=1000,
                                     warmup_steps=warmup_steps,
                                     output_path=self.cross_encoder_path)

    def label_set(self):
        """Use the fine-tuned model to predict the labels for the given dataset"""
        self.load_model(from_file=True)
        all_silver_scores = self.cross_encoder_model.predict(self.all_silver_data)
        # All model predictions should be between [0,1]
        assert all(0.0 <= score <= 1.0 for score in all_silver_scores)
        all_binary_silver_scores = [1 if score >= 0.5 else 0 for score in all_silver_scores]
        if self.task == 'classification':
            sentences_and_score = [(row[0], row[1], score)
                                   for row, score in zip(self.all_silver_data, all_binary_silver_scores)]
        elif self.task == 'regression':
            sentences_and_score = [(row[0], row[1], score)
                                   for row, score in zip(self.all_silver_data, all_silver_scores)]
        filename = str.replace(self.silver_set_path, ".tsv", "__CE.tsv")
        with open(filename, 'w', encoding='utf') as fout:
            fout.write("sentence_1\tsentence_2\tscore")
            for sss in sentences_and_score:
                fout.write(f"\n{sss[0]}\t{sss[1]}\t{sss[2]}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'Cross-Encoder fine-tuning')
    parser.add_argument("--dataset_name", required=True, type=str, help="Name of the dataset",
                        choices=["disney", "mrpc", "sts"])
    args = parser.parse_args()
    if args.dataset_name == "disney":
        raise NotImplementedError("CrossEncoder fine-tuning not implemented for DISNEY dataset")
    elif args.dataset_name == "mrpc":
        silver_set_path = "../data/MRPC/silver_set_classification0.5_cosine.tsv"
        dev_set_path = "../data/MRPC/dev_set.tsv"
        test_set_path = "../data/MRPC/test_set.tsv"
        task = "classification"
    elif args.dataset_name == "sts":
        silver_set_path = "../data/STS/silver_set_regression_cosine.tsv"
        dev_set_path = "../data/STS/dev_set.tsv"
        test_set_path = "../data/STS/test_set.tsv"
        task = "regression"
    fine_tuner = CrossEncoderFineTuner(silver_set_path=silver_set_path, dev_set_path=dev_set_path,
                                       dataset_name=str.upper(args.dataset_name), task=task)
    fine_tuner.fine_tune()
    fine_tuner.label_set()
