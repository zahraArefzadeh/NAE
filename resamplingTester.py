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
from ReverseAdasyn import REVERSEADASYN

from NearestNeighboursExactDenoise import NearestNeighboursExactDenoise
from NnedAdasynEnn import NnedAdasynEnn
from EnnAdasynNned import EnnAdasynNned
from sklearn.preprocessing import MinMaxScaler
random_state=10
# Define the dictionary mapping oversampling approach names to their respective functions
oversampling_methods = {
    'ADASYN': ADASYN,
    'SMOTE': SMOTE,
    'Borderline-SMOTE': BorderlineSMOTE,
    'SVMSMOTE': SVMSMOTE,
    'K-Means SMOTE': KMeansSMOTE,
    'SMOTE_ENN': SMOTE_ENN,
    'MSMOTE': MSMOTE,
    'Safe_Level_SMOTE':Safe_Level_SMOTE,
    'SOI_CJ': SOI_CJ,
    'KernelADASYN': KernelADASYN,
    'A_SUWO': A_SUWO,
    'Gaussian_SMOTE': Gaussian_SMOTE,
    'cluster_SMOTE':cluster_SMOTE
}

oversampling_methods={
    'NnedAdasynEnn':NnedAdasynEnn(),
    'ReverseAdasyn':REVERSEADASYN(),
    'ADASYNENN':ADASYNENN(),
    'ENNADASYNENN':EnnAdasynEnn(),
    'AdasynNned':AdasynNned(),
    'EnnAdasynNned':EnnAdasynNned()
    }
# Define a function to apply oversampling, train the classifier, and evaluate it
def train_and_evaluate(filename,X_train, y_train, X_test, y_test , oversampler_class, **kwargs):
    # Apply oversampling
    n_min_samples = np.sum(y_train[y_train == 1])
    n_neighbors = min(n_min_samples - 1,5)
    #--------------------------------------------------------------------------
    # if('n_neighbors' in oversampler_class().get_params()):
    #     oversampler = oversampler_class(n_neighbors=n_neighbors, **kwargs)
        
    # if('cluster_balance_threshold' in oversampler_class().get_params()):
    #     oversampler = oversampler_class(cluster_balance_threshold=0.01, **kwargs)
    # else:
    #     oversampler = oversampler_class( **kwargs)
    oversampler=oversampler_class
    # try:
    X_res, y_res = oversampler.fit_resample(X_train, y_train)
    # except:
    #     print(filename,oversampler_class)
    #     X_res, y_res = X_train, y_train
    #--------------------------------------------------------------------------
    # Train the classifier
    classifier = RandomForestClassifier(random_state=random_state)
    classifier.fit(X_res, y_res)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test)
    # y_pred_proba = classifier.predict_proba(X_test)[:, 1]  # For AUC, we need probability scores
    
    # Calculate AUC and F1 scores
    auc_score = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return auc_score, f1

# Path to your directory containing the .dat and .mat files
directory_path = 'BinaryDatasets/include'
#------------------------------------------------------------------------------
# Initialize DataFrames to hold the F1 and AUC results
f1_scores_df = pd.DataFrame(columns=['Dataset'] + list(oversampling_methods.keys()))
auc_scores_df = pd.DataFrame(columns=['Dataset'] + list(oversampling_methods.keys()))
#------------------------------------------------------------------------------
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
    #--------------------------------------------------------------------------
    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X=np.array(X).astype(float)
    y=np.array(y).astype(int)
    class_1_cout=np.sum(y)
    class_0_cout=y.shape[0]- class_1_cout
    #--------------------------------------------------------------------------
    #class 1 is always minority class
    if(class_1_cout>class_0_cout):
        y=1-y
    #--------------------------------------------------------------------------
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    #--------------------------------------------------------------------------
    #Normalization
    scaler = MinMaxScaler()
    # X_train=scaler.fit_transform(X_train)
    # X_test=scaler.transform(X_test)
    #--------------------------------------------------------------------------
    # Initialize dictionaries to hold scores for the current dataset
    f1_scores = {'Dataset': filename}
    auc_scores = {'Dataset': filename}
    #--------------------------------------------------------------------------
    # Iterate through each oversampling method and evaluate
    for name, method in oversampling_methods.items():
        # X_res, y_res = train_and_evaluate(X_train, y_train, method_class, random_state=42)
        auc_, f1_ = train_and_evaluate(filename,X_train, y_train, X_test, y_test , method, random_state=random_state)
        f1_scores[name] = f1_
        auc_scores[name] = auc_
    #--------------------------------------------------------------------------
    # Append the scores to the DataFrames
    f1_scores_df = f1_scores_df.append(f1_scores, ignore_index=True)
    auc_scores_df = auc_scores_df.append(auc_scores, ignore_index=True)
#------------------------------------------------------------------------------
# Excel file path to write the results
excel_output_path = 'Results-ourMethods-notNormalized.xlsx'
#------------------------------------------------------------------------------
# Write the DataFrames to separate sheets in the Excel file
with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
    f1_scores_df.to_excel(writer, sheet_name='F1 Scores', index=False)
    auc_scores_df.to_excel(writer, sheet_name='AUC Scores', index=False)
