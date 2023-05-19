import os
import numpy as np
import pandas as pd
import json
from scipy.stats import wilcoxon

SURVEY_DATA_PATH = "D:\\UNIVERSITA\\KTH\\THESIS\\ProjectCode\\data\\SURVEY"
BASELINE_IDENTIFIERS = "groups_{}_baseline_{}"
FT_IDENTIFIERS = "groups_{}_finetuned_{}"

# TODO - I confused intra and inter cluster. They should be switched!!!!

def analysis(analysis_type, n_questions, alternative):
    # mapping from RELATIVE question number (from 1 to n_questions, based on the set of questions associated to baseline/finetuned model)
    # to the absolute question number (in the survey, from 1 to 40)
    baseline_question_identifiers = [BASELINE_IDENTIFIERS.format(analysis_type, number) for number in range(1, n_questions + 1)]
    ft_question_identifiers = [FT_IDENTIFIERS.format(analysis_type, number) for number in range(1, n_questions + 1)]
    # inter_baseline2question_number = dict(filter(lambda x: x[0].startswith("groups_inter_cluster_baseline"),
    #                                               question_type2question_number.items()))
    # inter_ft2question_number = dict(filter(lambda x: x[0].startswith("groups_inter_cluster_finetuned"),
    #                                         question_type2question_number.items()))
    # inter_baseline2question_number = dict(sorted(inter_baseline2question_number.items()))
    # inter_ft2question_number = dict(sorted(inter_ft2question_number.items()))

    set_answers_baseline = answers_df[baseline_question_identifiers]
    set_answers_ft = answers_df[ft_question_identifiers]

    set_answers_baseline = np.concatenate([row.values for _, row in set_answers_baseline.iterrows()], axis=-1)
    set_answers_ft = np.concatenate([row.values for _, row in set_answers_ft.iterrows()], axis=-1)

    # Perform Wilcoxon signed-rank test with alternative='greater'
    # we set alternative=greater in order to test if FT group is greater than BASELINE group
    res = wilcoxon(set_answers_ft, set_answers_baseline, alternative=alternative)
    statistic, p_value = res.statistic, res.pvalue
    # I switched intra and inter analysis by mistake
    analysis_type = 'intra' if analysis_type=='inter_cluster' else 'inter' if analysis_type=='intra_cluster' else analysis_type
    alternative = 'larger' if alternative=='greater' else 'lower'
    print(f"Wilcoxon signed-rank test {analysis_type}")
    print("------------------------")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    # if the pvalue is lower than the significance level (usually 5%),
    # we reject the null hypothesis and conclude that FT median is greater/lower than BASELINE median!
    # greater/lower based on the alternative argument (greater for intra cluster analysis, lower otherwise)
    if p_value < 0.05:
        print(f"The difference between FT and BASELINE is statistically significant and FT has {alternative} scores.")
    else:
        print(f"The difference between FT and BASELINE is not statistically significant or BASELINE has {alternative} scores.")


# choice = int(input("Insert option: \n"
#                "1 - Intra cluster analysis\n"
#                "2 - Inter cluster analysis\n"
#                "3 - Outlier analysis...\n"
#                ""))
# assert choice in [1, 2, 3], f"Invalid choice. Valid options are 1, 2, 3. Inserted {choice} instead"

survey_question_mapping_path = os.path.join(SURVEY_DATA_PATH, "QUESTIONS", "map_group2question.json")
answers_path = os.path.join(SURVEY_DATA_PATH, "answers.txt")

answers_df = pd.read_csv(answers_path, header=None)
answers_df = answers_df.iloc[:, 1:]  # remove first column

question_type2question_number = json.load(open(survey_question_mapping_path, "r"))

answers_df.columns = list(question_type2question_number.keys())

analysis('inter_cluster', 10, 'greater')
print("----------------------------------------\n\n")
analysis('intra_cluster', 5, 'less')
print("----------------------------------------\n\n")
analysis('outliers', 5, 'less')
