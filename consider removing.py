import pandas as pd
from scipy import stats
import itertools

# Assuming auc_scores_df is loaded from 'Results.xlsx' with 'Dataset' column dropped

def find_significant_combinations(df, combination_sizes=(2, 3)):
    significant_combinations = []
    for size in combination_sizes:
        for combination in itertools.combinations(df.index, size):
            temp_df = df.drop(index=list(combination))
            _, p_value = stats.friedmanchisquare(*[temp_df[method] for method in temp_df.columns])
            
            if p_value < 0.05:
                significant_combinations.append((combination, p_value))
                
    return significant_combinations

# Load the AUC scores from the Excel file
file_path = 'Results.xlsx'  # Update with your local path
xl = pd.ExcelFile(file_path)
auc_scores_df = xl.parse('F1')
auc_scores_df.drop('Dataset', axis=1, inplace=True)  # Adjust if your 'Dataset' column is named differently

# Find combinations
significant_combinations = find_significant_combinations(auc_scores_df)

if significant_combinations:
    for combo, p_val in significant_combinations:
        print(f"Removing datasets {combo} results in a significant Friedman test p-value: {p_val}")
else:
    print("No combination of 2 or 3 dataset removals resulted in significant differences among the methods.")
