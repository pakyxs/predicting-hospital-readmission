from flask import Flask, render_template
import pandas as pd
import numpy as np
import os
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna

app = Flask(__name__)


# Flask route to display index
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to display the wine data
@app.route('/diabetes-data')
def get_diabetes_data():
    # Read the file
    workdir = os.getcwd()  # Get the current working directory
    file_path = os.path.join(workdir, './dataset/diabetic_data.csv') 

    original_df = pd.read_csv(file_path)

    df = original_df.copy(deep=True)
    return df.to_json()


# Flask route to display the DataFrame with clustering performed
@app.route('/prediction/<input>}')
def prediction(input):

    diabetes_data = get_diabetes_data()
    
    # Transform JSON datato DataFrame
    df = pd.read_json(diabetes_data)

        # Drop features
    def drop_features(df):

        df = df.drop(['weight', 'payer_code', 'medical_specialty', 'examide', 'citoglipton'], axis=1)

        # drop bad data with 3 '?' in diag
        drop_ID = set(df[(df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')].index)

        # drop died patient data which 'discharge_disposition_id' == 11 | 19 | 20 | 21 indicates 'Expired'
        drop_ID = drop_ID.union(set(df[(df['discharge_disposition_id'] == 11) | (df['discharge_disposition_id'] == 19) | \
                                    (df['discharge_disposition_id'] == 20) | (df['discharge_disposition_id'] == 21)].index))

        # drop 3 data with 'Unknown/Invalid' gender
        drop_ID = drop_ID.union(df['gender'][df['gender'] == 'Unknown/Invalid'].index)

        # create a set of indexes to drop the IDs that are not required
        new_ID = list(set(df.index) - set(drop_ID))

        df = df.iloc[new_ID]

        return df

    # Encode data
    def encode(df):
        """Encode features.

        Args:
            df (pd.DataFrame): The DataFrame containing the columns you need to encode.

        Returns:
            pd.DataFrame: The DataFrame with encoded features.
        """
        df['readmitted'] = df['readmitted'].replace({'>30': 1, '<30': 1, 'NO': 0}) # Encode readmitted column
        df['race'] = df['race'].replace({'Asian':0, 'AfricanAmerican':1, 'Caucasian': 2, 'Hispanic': 3, 'Other': 4, '?': 4})# Encode race column
        df['A1Cresult'] = df['A1Cresult'].replace({'None': -99, '>8': 1, '>7': 1, 'Norm': 0})# Encode A1Cresult column
        df['max_glu_serum'] = df['max_glu_serum'].replace({'>200': 1, '>300': 1, 'Norm': 0, 'None': -99})# Encode max_glu_serum column
        df['change'] = df['change'].replace({'No': 0, 'Ch': 1})# Encode change column
        df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})# Encode gender column
        df['diabetesMed'] = df['diabetesMed'].replace({'Yes': 1, 'No':0})# Encode diabetesMed column
        # Encode age column
        age_dict = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
                '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
        df['age'] = df['age'].replace(age_dict)
        df['age'] = df['age'].astype('int64')
        return df


    #  Classify Diagnoses by ICD-9
    def classify_diagnoses_by_icd9(df):
        """Classifies diagnoses by ICD-9 codes and creates additional features.

        Args:
            df (pd.DataFrame): The DataFrame containing the diagnosis columns.

        Returns:
            pd.DataFrame: The DataFrame with classified diagnoses and added features.
        """

        def classify_diag_column(column_name):
            df.loc[df[column_name].str.contains('V|E', na=False), column_name] = 0
            df[column_name] = df[column_name].replace('?', -1)
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

            df[column_name].replace(
                to_replace=range(1, 1000),
                value=pd.cut(range(1, 1000), bins=17, labels=range(1, 18)),
                inplace=True
            )

        for column_name in ['diag_1', 'diag_2', 'diag_3']:
            classify_diag_column(column_name)   

        return df


    # Function to encode medicines
    def process_medication_data(df, medications):
        """
        Processes medication data in a DataFrame, creating columns for "Up", "Down", and total count.

        Args:
            df (pd.DataFrame): The DataFrame containing medication data.
            medications (list): A list of medication names.

        Returns:
            pd.DataFrame: The processed DataFrame with new medication columns.
        """

        for med in medications:
            df[med] = df[med].replace({
                "Up": 1, "Down": 1, "Steady": 0, "No": 0,
            }).fillna(0)

        df['num_med_taken'] = df[[med for med in medications]].sum(axis=1)
        return df

    medicine = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 
                'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide',
                'metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin',
                'troglitazone', 'tolbutamide', 'acetohexamide']

    # Standarize numeric features
    def standardize(df, numeric_cols):
        """Standardizes numeric columns in a DataFrame, preserving non-numeric values.

        Args:
            df (pd.DataFrame): The DataFrame containing the numeric columns.
            numeric_cols (list): A list of numeric column names.

        Returns:
            pd.DataFrame: The DataFrame with standardized numeric columns.
        """

        for col in numeric_cols:
            try:
                # Attempt to convert to numeric and standardize
                df[col] = pd.to_numeric(df[col])
                df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=1)
            except:
                # If conversion fails, keep original values
                pass

        # Remove outliers based on z-scores (only for numeric values)
        df = df[np.abs(zscore(df[numeric_cols])) < 3] 

        return df
    # Apply function drop features
    df = drop_features(df)
    # Apply function to encode df
    df = encode(df)
    # Apply the function to classify diagnoses by ICD-9
    df = classify_diagnoses_by_icd9(df) 
    # Apply function to encode medicines
    df_medicine = process_medication_data(df, medicine)
    # Apply the function to standarize numeric columns
    numeric_columns = ['race', 'age', 'time_in_hospital', 'num_medications', 'number_diagnoses',
                    'num_med_taken', 'number_inpatient', 'number_outpatient', 'number_emergency',
                    'num_procedures', 'num_lab_procedures']

    df_numeric = standardize(df, numeric_columns)

    # Define target and features
    features = df.drop('readmitted', axis=1)
    target = df['readmitted']

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2)

    # Train model with best hyperparameters
    best_model = xgb.XGBClassifier(n_estimators = 140, max_depth= 8, learning_rate = 0.035962401064329685)
    best_model.fit(X_train, Y_train)

    # Define function to predict imput data
    def pred_imput(input):
    
        # Remove brackets and split the string into individual elements
        elements = input.replace(' ', '').replace('[', '').replace(']', '').split(',')

        # Convert each non-empty element to float and create the array
        patient_data_array = []
        for element in elements:
            if element.strip():  # Check if the element is not empty after stripping whitespace
                try:
                    patient_data_array.append(float(element))
                except ValueError:
                    pass  # Skip if conversion to float fails
        # Convert the list to a numpy array
        patient_data_array = np.array(patient_data_array)

        if len(patient_data_array) == 0:
            # Handle the case where no valid elements were found
            print("No valid elements found in the input data.")
            return None
        
        # Reshape the array
        patient_data_reshaped = patient_data_array.reshape(1, -1)
        
        # Make the prediction
        pred = best_model.predict(patient_data_reshaped)
        return pred
    
    output = pred_imput(input)

    if output[0] == 1:
        msg = 'The patient will likely be readmitted to the institution.'
        return msg
    else:
        msg = 'The patient will not be readmitted to the institution.'
        return msg




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')