import pandas as pd
from sklearn.model_selection import train_test_split
import logging

def load_and_preprocess_data(files):
    dataframes = []
    for file in files:
        try:
            logging.info(f"Loading file: {file.name}")
            df = pd.read_csv(file)
            if df.empty:
                raise ValueError("The uploaded file is empty.")
            if df['churn'].dtype == 'object':
                mapping = {'No': 0, 'Yes': 1, '0': 0, '1': 1}
                df['churn'] = df['churn'].map(mapping)
                if df['churn'].isnull().any():
                    raise ValueError("The target variable contains unknown values that could not be mapped to binary.")
            
            dataframes.append(df)
        except pd.errors.EmptyDataError:
            raise ValueError("No columns to parse from file.")
        except Exception as e:
            raise ValueError(f"An error occurred while reading the file: {str(e)}")
    
    if len(dataframes) == 0:
        raise ValueError("No valid dataframes loaded.")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    X = combined_df.iloc[:, :-1]
    y = combined_df['churn']

    if y.nunique() != 2:
        raise ValueError("The target variable is not binary. Please provide a binary target variable for classification.")

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
