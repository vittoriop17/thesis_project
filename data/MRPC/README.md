# Original Data
The original dataset can be downloaded from [MRPC dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52398)

The files 
- msr_paraphrase_data.txt
- msr_paraphrase_test.txt
- msr_paraphrase_train.txt

come from the original dataset.
The training dataset won't be directly used for training the model.
Instead, it will be only used to extract validation data (25% of the training set).
The training dataset will be constructed automatically using the **SILVER SET CONSTRUCTION TECHNIQUE**.

The files
- train_set.tsv
- dev_set__scenario_0.tsv

are both extracted from msr_paraphrase_train.txt using a 90-10 split. Both datasets contain ground truth labels.
In this way, it is possible to fine-tune the models with real labels and compare it with the automatic training set construction technique

# Silver Set Construction
The Silver Set is built using the _extract_sentence_pairs_ and _label_sentence_pairs_ functions 
(check module silver_set_construction.py)
The Silver Set is used for training, and it should differ from the original training set to a certain extent, since
the objective of the research is to test the validity of automatic set construction.

# Silver set - Scenario 1
The silver set for scenario 1 is obtained using the Cross Encoder after fine-tuning.
It has to be extracted using the script in the **models** package (_cross_encoder_fine_tuning_)
The silver set extracted is named with __CE as suffix 
