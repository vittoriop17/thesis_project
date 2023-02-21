# Silver Set Construction
Set of steps:
1. filtering out short and long sentence (>3 words, <128 words)
2. reduce set size to MAX_SENTENCES (=25_000)
3. split sentences into train, dev, test sentences (60-20-20)
4. create training, dev, test silver sets

N.B.: dev_set.tsv and test_set.tsv are built using the SilverSetConstruction technique, since there are
no labeled data for this dataset. The assessment of the fine-tuning is made through fake labels...

# Alternative to dev and test sets
Use STS dev and test splits. In this way, we can rely on true labels. 
