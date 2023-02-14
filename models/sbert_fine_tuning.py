from torch.utils.data import DataLoader
from sentence_transformers import models, losses, util, LoggingHandler, SentenceTransformer, SentencesDataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator, \
    LabelAccuracyEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
from zipfile import ZipFile
import logging
import csv
import sys
import torch
from utils.utils import *
import math
import gzip
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sentence_transformers.losses import SoftmaxLoss, MSELoss, MultipleNegativesRankingLoss, CosineSimilarityLoss, ContrastiveLoss
from utils import *

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


class SbertFineTuning:
    def __init__(self, silver_set_path, dev_set_path, test_set_path, dataset_name, scenario, loss_type,
                 evaluator_type, batch_size: int = 16, num_epochs: int = 1, max_seq_length: int = 128,
                 evaluation_only: bool = False, bi_encoder_path=None, base_model='bert-base-uncased',
                 task='classification', strategy='quantile', n_bins=5, **kwargs):
        """

        :param silver_set_path:
        :param dev_set_path:
        :param test_set_path:
        :param dataset_name:
        :param scenario: int. {1, 2}
        :param evaluation_only:
        :param bi_encoder_path:
        :param base_model: str. Default 'bert-base-uncased'.
        :param task: str. {'classification', 'regression'}. Default 'classification'
         Argument added in order to tackle the STS problem in two possible ways:
         as a regression task or as a classification task. In this case, a kbins_discretization is required
        """
        self.valid_losses_dict = {
            'softmax': SoftmaxLoss,  # for multilabel classification (similarity with discretized scores: STS)
            'cosine': CosineSimilarityLoss,  # for regression (similarity: STS)
            # 'mse': MSELoss,  # for regression (similarity: STS) -- NOT VALID
            'multiple_neg_ranking': MultipleNegativesRankingLoss,  # for binary classification (paraphrase: MRPC)
            'contrastive': ContrastiveLoss  # for binary classification (paraphrase: MRPC)
        }
        self.valid_evaluators_dict = {
            'binary': BinaryClassificationEvaluator,
            'regression': EmbeddingSimilarityEvaluator,
            'multilabel_accuracy': LabelAccuracyEvaluator
        }
        os.chdir("models") if os.getcwd().endswith('ProjectCode') else None
        logging.info("Start checking argument...")
        self.sanity_check(dataset_name, scenario, bi_encoder_path, evaluation_only, silver_set_path, task, loss_type,
                          evaluator_type, strategy)
        logging.info("...check done")
        self.scenario = scenario
        self.dataset_name = dataset_name
        self.dev_set_path = dev_set_path
        self.test_set_path = test_set_path
        self.base_model = base_model
        self.task = task
        # Initialize loss and evaluator functions
        self.loss = self.valid_losses_dict[str.lower(loss_type)]
        self.loss_type = loss_type
        self.evaluator = self.valid_evaluators_dict[evaluator_type]
        self.test_evaluator = self.valid_evaluators_dict[evaluator_type]
        self.evaluator_type = evaluator_type
        # arguments for kbins_discretizer (used only for STS dataset solved as classification task
        self.strategy = strategy if dataset_name == 'STS' and task == 'classification' else None
        self.n_bins = n_bins if dataset_name == 'STS' and task == 'classification' else None
        self.kbins_discretizer = None  # value assigned only if dataset_name = STS and task = classification
        # params taken from AugSBERT paper
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_seq_length = max_seq_length
        self.use_cuda = torch.cuda.is_available()
        self.silver_set_path = silver_set_path
        # used to save the fine-tuned model
        self.bi_encoder_path = os.path.join(dataset_name, f"scenario_{scenario}",
                                            "bi_encoder" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) \
            if evaluation_only == False else bi_encoder_path
        self.save_arguments()
        self.silver_data = []
        self.silver_scores = []
        self.binary_silver_scores = []
        self.bi_encoder_model = None
        self.evaluation_only = evaluation_only
        self.softmax_loss = None
        self.test_evaluator_flag = False

    def sanity_check(self, dataset_name, scenario, bi_encoder_path,
                     evaluation_only, silver_set_path, task, loss_type, evaluator_type, strategy):
        assert str.upper(dataset_name) in ['MRPC', 'STS', 'DISNEY'], \
            f"Invalid value for argument 'dataset_name'!, Expected one of these: MRPC, STS, DISNEY. Found {dataset_name}"
        assert scenario in [1, 2], f"Invalid value for argument 'scenario'. Expected 1 or 2. Found: {scenario}"
        assert bi_encoder_path is not None if evaluation_only else True, \
            "Argument 'bi_encoder_path' required if 'evaluation_only==True'!"
        assert os.path.exists(bi_encoder_path) if evaluation_only else True, \
            f"Pre-trained bi-encoder model not found! Provided path: {bi_encoder_path}"
        assert os.path.exists(silver_set_path), f"Target dataset not found. Given path: {silver_set_path}"
        assert task in ['classification', 'regression'], f"Invalid value for argument 'task'. Expected 'classification'" \
                                                         f"or 'regression'. Found: {task}"
        assert task == 'classification' if dataset_name == 'MRPC' else True, f"Invalid value for argument 'task'." \
                                                                             f" Expected 'classification' when " \
                                                                             f"'dataset_name'==MRPC." \
                                                                             f" Found: {task}"
        loss_type = str.lower(loss_type)
        assert loss_type in self.valid_losses_dict.keys(), f"Invalid value for argument 'loss'. " \
                                                           f"Expected one of the following: {self.valid_losses_dict.keys()}. " \
                                                           f"Found: {loss_type}"
        assert strategy in ['quantile', 'uniform', None], f"Invalid value for argument 'strategy'. " \
                                                          f"Expected one of the following: 'quantile', 'uniform', None" \
                                                          f"Found: {strategy}"
        assert evaluator_type in self.valid_evaluators_dict.keys(), \
            f"Invalid value for argument 'evaluator_type'. " \
            f"Expected one of the following: {self.valid_evaluators_dict.keys()}. " \
            f"Found: {evaluator_type}"
        assert loss_type in ['cosine'] if dataset_name == 'STS' and task == 'regression' else True, \
            f"Invalid value for argument 'loss'. Expected one of these: 'cosine' when " \
            f"using STS dataset and regression task. Found: {loss_type}"
        assert loss_type == 'softmax' if dataset_name == 'STS' and task == 'classification' else True, \
            f"Invalid value for argument 'loss'. Expected 'softmax' when using STS dataset and " \
            f"classification task. Found: {loss_type}"
        assert loss_type in ["multiple_neg_ranking", "contrastive"] if dataset_name == 'MRPC' else True, \
            f"Invalid value for argument 'loss'. Expected 'multiple_neg_ranking' or 'contrastive' when using MRPC " \
            f"dataset. Found: {loss_type}"
        assert evaluator_type == 'binary' if dataset_name == 'MRPC' else True, \
            f"Invalid value for argument 'evaluator_type'. Expected 'binary' when using MRPC " \
            f"dataset. Found: {evaluator_type}"
        assert evaluator_type == 'multilabel_accuracy' if dataset_name == 'STS' and task == 'classification' else True, \
            f"Invalid value for argument 'evaluator_type'. Expected 'multilabel_accuracy' when using STS " \
            f"dataset and classification task. Found: {evaluator_type}"
        assert evaluator_type == 'regression' if dataset_name == 'STS' and task == 'regression' else True, \
            f"Invalid value for argument 'evaluator_type'. Expected 'regression' when using STS " \
            f"dataset and regression task. Found: {evaluator_type}"
        assert loss_type == 'softmax' if evaluator_type == 'multilabel_accuracy' else True, \
            f"Invalid value for argument 'loss_type'. Expected 'softmax' when using " \
            f"multilabel_accuracy as metric. Found: {loss_type}"

    def save_arguments(self):
        try:
            os.makedirs(self.bi_encoder_path)
        except:
            pass
        dictionary = {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "dataset_name": self.dataset_name,
            "task": self.task,
            "strategy": self.strategy,
            "base_model": self.base_model,
            "scenario": self.scenario,
            "n_bins": self.n_bins,
            "max_seq_length": self.max_seq_length,
            "loss": self.loss.__name__,
            "evaluator": self.evaluator.__name__
        }
        filepath = os.path.join(self.bi_encoder_path, "args.json")
        save_args_to_json(dictionary, filepath)

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
            print("Read silver set for scenario 1 not implemented!")
            exit(-10)
        # if task == regression, then the scores are already stored inside self.silver_scores
        # if task == classification, then the scores are discretized. Stored inside self.silver_scores, again
        if self.dataset_name == 'STS' and self.task == 'classification':
            self.prepare_sts_silver_data()

    def prepare_sts_silver_data(self):
        """
        Function called only when using STS data.
        :return:
        """
        self.kbins_discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
        self.silver_scores = self.kbins_discretizer.fit_transform(np.reshape(self.silver_scores, (-1, 1)))
        self.silver_scores = self.silver_scores.astype('int64')

    def prepare_evaluator(self, dev=True, softmax_loss=None):
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
                if self.dataset_name == 'STS':
                    labels.append(float(row['score']))
                else:
                    labels.append(int(float(row['score'])))  # Note: binary label
        if self.dataset_name == 'STS' and self.task == 'classification':
            labels = self.kbins_discretizer.transform(np.reshape(labels, (-1, 1))).reshape(-1)
            if self.evaluator_type == 'multilabel_accuracy':
                data_ = list(InputExample(texts=[s1, s2], label=score)
                             for (s1, s2, score) in zip(sentences1, sentences2, labels))
                data_loader = DataLoader(data_, shuffle=True, batch_size=self.batch_size)
                self.evaluator = self.evaluator(data_loader, name='Dev set' if dev else 'test set',
                                                softmax_model=softmax_loss)
        else:
            self.evaluator = self.evaluator(sentences1, sentences2, labels, name='Dev set' if dev else 'test set')

    def prepare_test_evaluator(self):
        if self.test_evaluator_flag:
            return
        logging.info(f"Preparing test evaluator")
        sentences1 = []
        sentences2 = []
        labels = []
        eval_data_path = self.test_set_path
        with open(eval_data_path, encoding='utf8') as fin:
            reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                sentences1.append(row['sentence_1'])
                sentences2.append(row['sentence_2'])
                if self.dataset_name == 'STS':
                    labels.append(float(row['score']))
                else:
                    labels.append(int(float(row['score'])))  # Note: binary label
        if self.dataset_name == 'STS' and self.task == 'classification':
            labels = self.kbins_discretizer.transform(np.reshape(labels, (-1, 1))).reshape(-1)
            if self.evaluator_type == 'multilabel_accuracy':
                data_ = list(InputExample(texts=[s1, s2], label=score)
                             for (s1, s2, score) in zip(sentences1, sentences2, labels))
                data_loader = DataLoader(data_, shuffle=True, batch_size=self.batch_size)
                self.test_evaluator = self.test_evaluator(data_loader, name='test set', softmax_model=self.softmax_loss)
        else:
            self.test_evaluator = self.test_evaluator(sentences1, sentences2, labels, name='test set')
        self.test_evaluator_flag = True

    def fine_tune_sbert(self):
        """
        Apply sbert fine-tuning, using the pre-loaded silver set
        :return:
        """
        logging.info(f"Fine tune bi-encoder: over labeled Silver Set ({self.dataset_name})")
        (train_dataloader, train_loss) = self.get_training_objectives()
        self.prepare_evaluator(dev=True, softmax_loss=train_loss)
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

    def get_training_objectives(self):
        """
        Return tuple with training objectives: (train_dataloader, train_loss)
        Check available losses at https://www.sbert.net/docs/package_reference/losses.html
        :return:
        """
        app_scores = self.binary_silver_scores if self.dataset_name == 'MRPC' else self.silver_scores
        train_data = list(InputExample(texts=[data[0], data[1]], label=score)
                          for (data, score) in zip(self.silver_data, app_scores))
        if self.loss_type == 'multiple_neg_ranking':  # remove all the sentences with score 0. See sbert multiple negative ranking loss
            train_data = list(filter(lambda x: x.label == 1, train_data))
            print(f"Number of training sentence pairs after filtering unrelated sentences: {len(train_data)}")
        if self.loss_type == 'softmax':
            train_dataset = SentencesDataset(train_data, self.bi_encoder_model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
            train_loss = self.loss(model=self.bi_encoder_model,
                                   sentence_embedding_dimension=768,
                                   num_labels=self.n_bins)
            self.softmax_loss = train_loss
        else:
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
            train_loss = self.loss(model=self.bi_encoder_model)
        return (train_dataloader, train_loss)

    def evaluate_sbert(self):
        logging.info("Starting evaluation on test data...\n")
        if self.loss_type == 'softmax':
            self.softmax_loss = self.loss(model=self.bi_encoder_model,
                                          sentence_embedding_dimension=768,
                                          num_labels=self.n_bins) if self.softmax_loss is None else self.softmax_loss
        self.prepare_test_evaluator()
        self.bi_encoder_model.evaluate(self.test_evaluator, output_path="tmp_test")


if __name__ == '__main__':
    silver_set_path = "../data/STS/silver_set_regression_cosine.tsv"
    dev_set_path = "../data/STS/dev_set.tsv"
    test_set_path = "../data/STS/test_set.tsv"
    # silver_set_path = "../data/MRPC/silver_set_classification0.5_cosine.tsv"
    # dev_set_path = "../data/MRPC/dev_set.tsv"
    # test_set_path = "../data/MRPC/test_set.tsv"

    fine_tuner = SbertFineTuning(silver_set_path=silver_set_path, dev_set_path=dev_set_path,
                                 test_set_path=test_set_path, dataset_name="STS", scenario=2, loss_type='softmax',
                                 task='classification', evaluator_type="multilabel_accuracy")
