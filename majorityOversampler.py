import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from crucio import SLS
from smote_variants import SMOTE_ENN,MSMOTE,Safe_Level_SMOTE,SOI_CJ,KernelADASYN,A_SUWO,Gaussian_SMOTE,cluster_SMOTE
from ADASYNENN import ADASYNENN
from AdasynNned import AdasynNned
from ENNADASYNENN import EnnAdasynEnn
from NearestNeighboursExactDenoise import NearestNeighboursExactDenoise
from NnedAdasynEnn import NnedAdasynEnn
from EnnAdasynNned import EnnAdasynNned
from sklearn.preprocessing import MinMaxScaler
random_state=10
oversampling_methods = {
    'ADASYN': ADASYN,
    'SMOTE': SMOTE,
    'Borderline-SMOTE': BorderlineSMOTE,
    'K-Means SMOTE': KMeansSMOTE,
    'cluster_SMOTE':cluster_SMOTE
}

def train_and_evaluate(oversampling_methods,X_train, y_train, X_test, y_test, **kwargs):
    # Apply oversampling
    n_min_samples = y_train[y_train == 1].shape[0]
    n_neighbors = min(n_min_samples - 1,5)
    y_pred_total=np.zeros((y_test.shape[0]))
    for name, method in oversampling_methods.items():
        if('n_neighbors' in method().get_params()):
            oversampler = method(n_neighbors=n_neighbors, **kwargs)
        
        if('cluster_balance_threshold' in method().get_params()):
            oversampler = method(cluster_balance_threshold=0.01, **kwargs)
        else:
            oversampler = method( **kwargs)
        X_res, y_res = oversampler.fit_resample(X_train, y_train)
        classifier = RandomForestClassifier(random_state=random_state)
        classifier.fit(X_res, y_res)
        # Make predictions on the test set
        y_pred = classifier.predict(X_test)
        y_pred_total+=y_pred
    
    print(y_pred_total)
    y_pred_total=((y_pred_total/n_neighbors)>0.5)*1
    # Calculate AUC and F1 scores
    auc_score = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return auc_score, f1

# Path to your directory containing the .dat and .mat files
directory_path = 'BinaryDatasets'

# Initialize DataFrames to hold the F1 and AUC results
f1_scores_df = pd.DataFrame(columns=['Dataset'] + list(oversampling_methods.keys()))
auc_scores_df = pd.DataFrame(columns=['Dataset'] + list(oversampling_methods.keys()))

# Iterate through each file in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    # Check the file extension and process accordingly
    if filename.endswith('.dat'):
        # Load .dat file as CSV
        df = pd.read_csv(file_path, header=None)
        # Convert the last column to numeric labels if it's categorical
        df.iloc[:, -1] = pd.factorize(df.iloc[:, -1])[0]
        
    elif filename.endswith('.mat'):
        # Load .mat file
        mat = loadmat(file_path)
        # Assume 'X' and 'y' are the variable names in the .mat file
        df = pd.DataFrame(np.hstack((mat['X'], mat['y'])))
    
    else:
        continue  # Skip files with other extensions

    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    class_1_cout=np.sum(y)
    class_0_cout=y.shape[0]- class_1_cout
    if(class_1_cout>class_0_cout):
        y=1-y

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = MinMaxScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    # Initialize dictionaries to hold scores for the current dataset
    f1_scores = {'Dataset': filename}
    auc_scores = {'Dataset': filename}
    
    # Iterate through each oversampling method and evaluate
    
    auc_, f1_ = train_and_evaluate(oversampling_methods,X_train, y_train, X_test, y_test , random_state=random_state)
    f1_scores['major'] = f1_
    auc_scores['major'] = auc_

    # Append the scores to the DataFrames
    f1_scores_df = f1_scores_df.append(f1_scores, ignore_index=True)
    auc_scores_df = auc_scores_df.append(auc_scores, ignore_index=True)

# Excel file path to write the results
excel_output_path = 'Results-baseMethods-Normalized.xlsx'

# Write the DataFrames to separate sheets in the Excel file
with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
    f1_scores_df.to_excel(writer, sheet_name='F1 Scores', index=False)
    auc_scores_df.to_excel(writer, sheet_name='AUC Scores', index=False)
