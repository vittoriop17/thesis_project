# Original Data
The original dataset can be downloaded from [STS dataset](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)

The files 
- sts-train.csv
- sts-dev.csv
- sts-test.csv

come from the original dataset.
The training dataset won't be directly used for training the model.
The training dataset will be constructed automatically using the **SILVER SET CONSTRUCTION TECHNIQUE**.

The files
- train_set.tsv
- dev_set.tsv
- test_set.tsv

are prepared from the aforementioned files. These 3 files (differently from the previous) contain only 3 columns:
- score
- sentence 1
- sentence 2

where the score has been normalized between 0 and 1 (/5)

# Silver Set Construction
The Silver Set is built using the _extract_sentence_pairs_ and _label_sentence_pairs_ functions 
(check module silver_set_construction.py)
The Silver Set is used for training, and it should differ from the original training set to a certain extent, since
the objective of the research is to test the validity of automatic set construction.

# Silver set - Scenario 1
The silver set for scenario 1 is obtained using the Cross Encoder after fine-tuning.
It has to be extracted using the script in the **models** package (_cross_encoder_fine_tuning_)
The silver set extracted is named with __CE as suffix 
