# Load a pre-trained CROSS-ENCODER (trained on STSb)
# Use the pre-trained model to label the target dataset (sentence-apirs)
# Fine-tune a pre-trained bi-encoder using the silver labels (fake labels)

from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder
from sentence_transformers import models, losses, util, LoggingHandler, SentenceTransformer
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
import logging
import csv
import torch
import math
import os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = "cross-encoder/stsb-roberta-base"
batch_size = 16
num_epochs = 1
max_seq_length = 128
use_cuda = torch.cuda.is_available()

###### Read Dataset ######
ssp_dataset_path = '../data/DISNEY/review_sentence_pairs.csv'
if not os.path.exists((ssp_dataset_path)):
    print(f"Target dataset not found. Expected path: {ssp_dataset_path}")
    exit(-1)

bi_encoder_path = 'output/bi-encoder/ssp_cross_domain_ ' + model_name.replace("/", "-") + '- ' + datetime.now().strftime \
    ("%Y-%m-%d_%H-%M-%S")

###### Bi-encoder (sentence-transformers) ######

logging.info("Loading bi-encoder model: {}".format(model_name))

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#####################################################
#
# Step 1: Load cross-encoder model pretrained over STSbenchmark
#
#####################################################

logging.info("Step 1: Loading cross-encoder model: {}".format(model_name))
cross_encoder = CrossEncoder(model_name)

##################################################################
#
# Step 2: Label SSp train dataset using cross-encoder model
#
##################################################################

logging.info("Step 2: Label SSP (target dataset) with cross-encoder: {}".format(model_name))

silver_data = []

with open(ssp_dataset_path, encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_NONE)
    for row in reader:
        silver_data.append([row['sentence1'], row['sentence2']])

silver_scores = cross_encoder.predict(silver_data)

# All model predictions should be between [0,1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)

binary_silver_scores = [1 if score >= 0.5 else 0 for score in silver_scores]

###########################################################################
#
# Step 3: Train bi-encoder (SBERT) model with SSP dataset - Augmented SBERT
#
###########################################################################

logging.info("Step 3: Train bi-encoder: {} over labeled SSP (target dataset)".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Loading BERT labeled SSP dataset")
ssp_train_data = list \
    (InputExample(texts=[data[0], data[1]], label=score) for (data, score) in zip(silver_data, binary_silver_scores))

train_dataloader = DataLoader(ssp_train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(bi_encoder)


# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the bi-encoder model
bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
               epochs=num_epochs,
               warmup_steps=warmup_steps,
               output_path=bi_encoder_path
               )


