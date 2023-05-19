import os.path
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import json

FOLDER_CLUSTERING_RESULTS = f"..\\results\\HDBSCAN"
N_INTER_CLUSTER_QUESTIONS = 10
N_INTRA_CLUSTER_QUESTIONS = 5
N_OUTLIERS_QUESTIONS = 5
FOLDER_SURVEY_SENTENCES = "..\\data\\SURVEY"
N_SENTENCES_PER_GROUP = 8

np.random.seed(1234567890)

# TODO - I confused intra and inter cluster. They should be switched!!!!

def save_questions(df_all_groups, folder):
    try:
        os.mkdir(folder)
    except:
        pass
    filename = "question_{}.txt"
    n_questions = df_all_groups.shape[0]
    group_name2question_number = dict()
    for question_number, group_idx in enumerate(np.random.permutation(np.arange(0, n_questions))):
        question_number += 1
        group_name = df_all_groups.index[group_idx]
        group_name2question_number[group_name] = question_number
        with open(os.path.join(folder, filename.format(question_number)), "w") as fp:
            sentences = str("\n\t").join(list(map(lambda x: f"sentence {x[0]}: \t{x[1]}", enumerate(list(df_all_groups.iloc[group_idx])))))
            fp.writelines(f"Question {question_number}:\n"
                          f"The following set of sentences is...\n"
                          f"\t{sentences}")
    json.dump(group_name2question_number, open(os.path.join(folder, "map_group2question.json"), "w"), indent=4)


def save_sentences_as_dict(list_df, filename):
    dict_sentences = {}
    for idx, current_df in enumerate(list_df):
        current_df = current_df.sort_values(by='dist_from_s', ascending=True)
        dict_sentences[f"{filename}_{idx + 1}"] = {f"sentence_{i}": s for i, s in enumerate(current_df.sentence.values)}
    fp = open(os.path.join(FOLDER_SURVEY_SENTENCES, filename), "w")
    json.dump(dict_sentences, fp, indent=4)
    fp.close()
    return dict_sentences


def get_closest_sentences_in_same_cluster(data, dist_mat, sentence, sid_to_id, n=10, c_id=None):
    # when c_id is specified, the closest sentences are taken from the cluster c_id
    # otherwise, the closest sentences are taken from the same cluster of the specified sentence
    c_id = data[data.sentence == sentence].cluster.values[0] if c_id is None else c_id
    data['dist_from_s'] = dist_mat[sid_to_id[hash(sentence)]]
    filtered_data = data[data.cluster == c_id]
    filtered_data.drop_duplicates(subset='sid', inplace=True)
    filtered_data.sort_values(by="dist_from_s", ascending=True, inplace=True)
    return filtered_data[:n]


def get_closest_sentences_from_n_different_clusters(data, dist_mat, sentence, sid_to_id, n=5):
    c_id = data[data.sentence == sentence].cluster.values[0]
    data['dist_from_s'] = dist_mat[sid_to_id[hash(sentence)]]
    sentences = [data.iloc[sid_to_id[hash(sentence)]]]
    filtered_data = pd.DataFrame.copy(data)
    filtered_data.drop_duplicates(subset='sid', inplace=True)
    filtered_data = filtered_data[filtered_data.cluster != 0]  # remove outliers
    for _ in range(n - 1):
        if len(filtered_data) == 0:
            print(f"No more data available. All the clusters have been filtered out. Try to reduce N")
            break
        filtered_data = filtered_data[filtered_data.cluster != c_id]
        filtered_data.sort_values(by="dist_from_s", ascending=True, inplace=True)
        sentences.append(filtered_data.iloc[0])
        c_id = filtered_data.cluster.values[0]  # store the cluster id of the closest sentence during this iteration
        # this value will then be used to filter out all the sentences belonging to this cluster
    return pd.concat(sentences, axis=1).T


df_baseline = pd.read_csv(os.path.join(FOLDER_CLUSTERING_RESULTS, "baseline.csv"))
df_finetuned = pd.read_csv(os.path.join(FOLDER_CLUSTERING_RESULTS, "finetuned.csv"))

df_baseline.sentence = [s.strip() for s in df_baseline.sentence]
df_finetuned.sentence = [s.strip() for s in df_finetuned.sentence]

df_baseline['sid'] = [hash(s) for s in df_baseline.sentence]
df_finetuned['sid'] = [hash(s) for s in df_finetuned.sentence]

cluster_distribution__baseline = df_baseline.groupby('cluster').agg('count').x1

outliers_baseline = df_baseline[df_baseline.cluster == 0]
outliers_finetuned = df_finetuned[df_finetuned.cluster == 0]

sid_common_outliers = set.intersection(set(outliers_baseline.sid), set(outliers_finetuned.sid))
print(f"Number of outliers in common: {len(sid_common_outliers)}")

random_sentences = []
max_cluster_id = max(df_finetuned.cluster) if max(df_finetuned.cluster) > max(df_baseline.cluster) else max(
    df_baseline.cluster)
df = df_finetuned if max(df_finetuned.cluster) > max(df_baseline.cluster) else df_baseline
df_2 = df_finetuned if max(df_finetuned.cluster) <= max(df_baseline.cluster) else df_baseline

for current_c_id in np.arange(1, max_cluster_id):  # 0 excluded (outliers)
    app_df = df[df.cluster == current_c_id]
    sents = np.random.choice(app_df.sentence, 5, False)
    random_sentences.extend(sents)

sents_ids = [hash(s) for s in random_sentences]
set_cluster_ids = set([row.cluster for _, row in df_2[df_2.sid.isin(sents_ids)].iterrows()])

cluster_ids_final_sents_finetuned = []
cluster_ids_final_sents_baseline = []
final_sentences = []
np.random.shuffle(random_sentences)
for s in random_sentences:
    c_baseline = df_baseline[df_baseline.sid == hash(s)].cluster.values[0]
    c_finetuned = df_finetuned[df_finetuned.sid == hash(s)].cluster.values[0]
    if c_baseline == 0 or c_finetuned == 0:
        continue
    if c_finetuned not in cluster_ids_final_sents_finetuned and c_baseline not in cluster_ids_final_sents_baseline:
        final_sentences.append(s)
        cluster_ids_final_sents_finetuned.append(c_finetuned)
        cluster_ids_final_sents_baseline.append(c_baseline)
        if len(final_sentences) == N_INTER_CLUSTER_QUESTIONS:
            break

# TODO
#  build the following structures:
#  1. G_m1,i, for i=1,2,3,...,N_INTER_CLUSTER_QUESTIONS
#     G_m1,i represents the GROUP OF SENTENCES extracted from the clustering results obtained with model 1 (baseline)
#     moreover, the index i represents the i-th sentence used the extraction above (i-th sentence in final_sentences)
#  2. G_m1,i for i=1,2,3,...,N_INTER_CLUSTER_QUESTIONS
#  REMEMBER: group of closest sentences to s_i
distance_matrix_df_baseline = pairwise_distances(X=np.concatenate((np.reshape(df_baseline.x1.values, (-1, 1)),
                                                                   np.reshape(df_baseline.x2.values, (-1, 1))),
                                                                  axis=-1))
distance_matrix_df_finetuned = pairwise_distances(X=np.concatenate((np.reshape(df_finetuned.x1.values, (-1, 1)),
                                                                    np.reshape(df_finetuned.x2.values, (-1, 1))),
                                                                   axis=-1))

sid_to_id_df_baseline = {row.sid: index for index, row in df_baseline.iterrows()}
sid_to_id_df_finetuned = {row.sid: index for index, row in df_finetuned.iterrows()}

# groups of sentences used for the evaluation of inter cluster results
groups_inter_cluster_baseline = [[] for _ in range(N_INTER_CLUSTER_QUESTIONS)]
groups_inter_cluster_finetuned = [[] for _ in range(N_INTER_CLUSTER_QUESTIONS)]

for idx, s in enumerate(final_sentences):
    groups_inter_cluster_baseline[idx] = get_closest_sentences_in_same_cluster(df_baseline, distance_matrix_df_baseline,
                                                                               s, sid_to_id_df_baseline,
                                                                               n=N_SENTENCES_PER_GROUP)
    groups_inter_cluster_finetuned[idx] = get_closest_sentences_in_same_cluster(df_finetuned,
                                                                                distance_matrix_df_finetuned, s,
                                                                                sid_to_id_df_finetuned,
                                                                                n=N_SENTENCES_PER_GROUP)

# TODO - prepare the INTRA-CLUSTER GROUPS (6 sentences per group) for 5 sentences and each distinct model outcome
# groups of sentences used for the evaluation of intra cluster results
groups_intra_cluster_baseline = [[] for _ in range(N_INTRA_CLUSTER_QUESTIONS)]
groups_intra_cluster_finetuned = [[] for _ in range(N_INTRA_CLUSTER_QUESTIONS)]

base_sentences_for_intra_cluster_eval = np.random.choice(final_sentences, size=N_INTRA_CLUSTER_QUESTIONS, replace=False)
for idx, s in enumerate(base_sentences_for_intra_cluster_eval):
    groups_intra_cluster_baseline[idx] = get_closest_sentences_from_n_different_clusters(df_baseline,
                                                                                         distance_matrix_df_baseline, s,
                                                                                         sid_to_id_df_baseline,
                                                                                         n=N_SENTENCES_PER_GROUP)
    groups_intra_cluster_finetuned[idx] = get_closest_sentences_from_n_different_clusters(df_finetuned,
                                                                                          distance_matrix_df_finetuned,
                                                                                          s, sid_to_id_df_finetuned,
                                                                                          n=N_SENTENCES_PER_GROUP)

# TODO - prepare the OUTLIER GROUPS (6 sentences per group) for 5 sentences and each distinct model outcome
# groups of sentences used for the evaluation of inter cluster results
groups_outliers_baseline = [[] for _ in range(N_OUTLIERS_QUESTIONS)]
groups_outliers_finetuned = [[] for _ in range(N_OUTLIERS_QUESTIONS)]

base_sentences_for_outliers_evaluation = np.random.choice(final_sentences, size=N_OUTLIERS_QUESTIONS, replace=False)

for idx, s in enumerate(base_sentences_for_outliers_evaluation):
    groups_outliers_baseline[idx] = get_closest_sentences_in_same_cluster(
        df_baseline, distance_matrix_df_baseline,
        s, sid_to_id_df_baseline, c_id=0,
        n=N_SENTENCES_PER_GROUP - 1).append(df_baseline.iloc[sid_to_id_df_baseline[hash(s)]])
    groups_outliers_finetuned[idx] = get_closest_sentences_in_same_cluster(df_finetuned,
                                                                           distance_matrix_df_finetuned, s,
                                                                           sid_to_id_df_finetuned, c_id=0,
                                                                           n=N_SENTENCES_PER_GROUP - 1).append(
        df_finetuned.iloc[sid_to_id_df_finetuned[hash(s)]])

# save all group of sentences:
df_groups_outliers_finetuned = pd.DataFrame.from_dict(
    save_sentences_as_dict(groups_outliers_finetuned, filename="groups_outliers_finetuned"))
df_groups_outliers_baseline = pd.DataFrame.from_dict(
    save_sentences_as_dict(groups_outliers_baseline, filename="groups_outliers_baseline"))
df_groups_intra_cluster_finetuned = pd.DataFrame.from_dict(
    save_sentences_as_dict(groups_intra_cluster_finetuned, filename="groups_intra_cluster_finetuned"))
df_groups_intra_cluster_baseline = pd.DataFrame.from_dict(
    save_sentences_as_dict(groups_intra_cluster_baseline, filename="groups_intra_cluster_baseline"))
df_groups_inter_cluster_finetuned = pd.DataFrame.from_dict(
    save_sentences_as_dict(groups_inter_cluster_finetuned, filename="groups_inter_cluster_finetuned"))
df_groups_inter_cluster_baseline = pd.DataFrame.from_dict(
    save_sentences_as_dict(groups_inter_cluster_baseline, filename="groups_inter_cluster_baseline"))


df_all_groups = pd.concat([
    df_groups_outliers_baseline,
    df_groups_outliers_finetuned,
    df_groups_inter_cluster_baseline,
    df_groups_inter_cluster_finetuned,
    df_groups_intra_cluster_baseline,
    df_groups_intra_cluster_finetuned
], axis=1).T  # .T so that we have the sentences along the columns and one group per row

save_questions(df_all_groups, os.path.join(FOLDER_SURVEY_SENTENCES, "QUESTIONS"))
# Overall, for each model, we should have:
# 10 questions for inter-cluster evaluation
# 5 questions for outlier detection
# 5 questions for intra-cluster evaluation
# Overall: 20 questions per model
