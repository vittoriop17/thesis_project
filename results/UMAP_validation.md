# UMAP - validation
Hyperparameters to fine-tune:
1. min_dist
2. n_components

Starting UMAP hyperparameters tuning (based on trustworthiness metric)...
Model params:	min_dist: 0.1,	n_components: 2,	trustworthiness: 0.8265411123723042
Model params:	min_dist: 0.1,	n_components: 3,	trustworthiness: 0.8706793237037531
Model params:	min_dist: 0.1,	n_components: 5,	trustworthiness: 0.9176468653416721
Model params:	min_dist: 0.5,	n_components: 2,	trustworthiness: 0.7620463454333682
Model params:	min_dist: 0.5,	n_components: 3,	trustworthiness: 0.8346050053044298
Model params:	min_dist: 0.5,	n_components: 5,	trustworthiness: 0.904142645092846
+Model params:	min_dist: 0.9,	n_components: 2,	trustworthiness: 0.7284791076687959
Model params:	min_dist: 0.9,	n_components: 3,	trustworthiness: 0.8002056955479884
Model params:	min_dist: 0.9,	n_components: 5,	trustworthiness: 0.8925850146520962

Starting UMAP hyperparameters tuning (based on trustworthiness metric)...
Model params:	min_dist: 0.1,	n_components: 2,	metric: cosine	trustworthiness: 0.8221799743310112
Model params:	min_dist: 0.1,	n_components: 2,	metric: euclidean	trustworthiness: 0.8116472653624447
Model params:	min_dist: 0.1,	n_components: 5,	metric: cosine	trustworthiness: 0.9177799152774996
Model params:	min_dist: 0.1,	n_components: 5,	metric: euclidean	trustworthiness: 0.9086877087089092
Model params:	min_dist: 0.9,	n_components: 2,	metric: cosine	trustworthiness: 0.7343744330937066
Model params:	min_dist: 0.9,	n_components: 2,	metric: euclidean	trustworthiness: 0.7301731603273167
Model params:	min_dist: 0.9,	n_components: 5,	metric: cosine	trustworthiness: 0.8925546584763304
Model params:	min_dist: 0.9,	n_components: 5,	metric: euclidean	trustworthiness: 0.8857036098313711
Best UMAP parameters and trustworthiness: {
    "min_dist": 0.1,
    "n_components": 5,
    "trustworthiness": 0.9177799152774996
}


Hyperparameters after hyp. tuning (including hyperparameters not fine-tuned):
1. min_dist: 0.1
2. n_components: 5
3. n_neighbors: 10
4. metric: cosine