import pandas as pd
from scipy import stats
import scikit_posthocs as sp

# Load the Excel file
file_path = 'Results.xlsx'  # Update with your local path
xl = pd.ExcelFile(file_path)

# Parse the 'AUC' sheet
auc_scores_df = xl.parse('F1')

# Drop any non-method columns if necessary, e.g., a 'Dataset' column
# If 'Dataset' column does not exist, remove or comment out the next line
auc_scores_df.drop('Dataset', axis=1, inplace=True)

# Perform the Friedman test to check for overall differences
friedman_statistic, friedman_p_value = stats.friedmanchisquare(*[auc_scores_df[method] for method in auc_scores_df.columns])

print(f'Friedman Test Statistic: {friedman_statistic}, P-value: {friedman_p_value}')

# If the Friedman test indicates significant differences, proceed with post-hoc analysis
if friedman_p_value < 0.05:  # Using 0.05 as a typical significance level
    posthoc_results = sp.posthoc_nemenyi_friedman(auc_scores_df)
    # Calculate mean ranks for each method
    mean_ranks = posthoc_results.mean(axis=0).sort_values(ascending=False)
    print('Post-hoc Test Rankings (Mean Ranks):\n', mean_ranks)
else:
    print('No significant differences found among the methods.')
