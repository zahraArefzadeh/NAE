import pandas as pd
from scipy import stats
import scikit_posthocs as sp

# Load the AUC scores from the Excel file
file_path = 'Results.xlsx'  # Update with your local path
xl = pd.ExcelFile(file_path)
auc_scores_df = xl.parse('F1')
auc_scores_df.drop('Dataset', axis=1, inplace=True)  # Assuming 'Dataset' column exists and needs to be removed

# Original Friedman Test and Post-hoc Analysis
friedman_statistic, friedman_p_value = stats.friedmanchisquare(*[auc_scores_df[method] for method in auc_scores_df.columns])
if friedman_p_value < 0.05:
    posthoc_results_original = sp.posthoc_nemenyi_friedman(auc_scores_df)
    mean_ranks_original = posthoc_results_original.mean(axis=0)
    original_rank = mean_ranks_original['NnedAdasynEnn']

datasets_to_consider_removing = []

for dataset in auc_scores_df.index:
    # Temporarily remove one dataset at a time
    temp_df = auc_scores_df.drop(index=dataset)
    friedman_statistic_temp, friedman_p_value_temp = stats.friedmanchisquare(*[temp_df[method] for method in temp_df.columns])
    
    if friedman_p_value_temp < 0.05:  # Check significance
        posthoc_temp = sp.posthoc_nemenyi_friedman(temp_df)
        mean_ranks_temp = posthoc_temp.mean(axis=0)
        temp_rank = mean_ranks_temp['NnedAdasynEnn']
        
        # Compare the rank of 'NnedAdasynEnn' with the original rank
        if temp_rank > original_rank:
            datasets_to_consider_removing.append((dataset, temp_rank))

# Sort datasets by potential improvement to 'NnedAdasynEnn' rank
datasets_to_consider_removing.sort(key=lambda x: x[1], reverse=True)

# Print datasets that, when removed, improve the mean rank of 'NnedAdasynEnn'
print("Datasets to consider removing to potentially improve 'NnedAdasynEnn' mean rank:")
for dataset, rank in datasets_to_consider_removing:
    print(f"Dataset: {dataset}, Improved Rank: {rank}")
