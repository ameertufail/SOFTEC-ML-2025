import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_train_data(file_path='train.csv'):
    """
    Load the train.csv file into a pandas DataFrame
    
    Parameters:
    -----------
    file_path : str, default='train.csv'
        Path to the training data CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the loaded data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_data_info(df):
    """
    Display basic information about the DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    """
    print("DataFrame shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe())

def convert_dates(df, date_columns=None):
    """
    Convert date columns to datetime format
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing date columns
    date_columns : list, default=None
        List of column names to convert to datetime.
        If None, will try to convert columns with 'DT' in their name
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with converted date columns
    """
    if date_columns is None:
        # Find columns that likely contain dates (those with 'DT' in name)
        date_columns = [col for col in df.columns if 'DT' in col]
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    for col in date_columns:
        try:
            df_copy[col] = pd.to_datetime(df_copy[col])
            print(f"Converted {col} to datetime")
        except Exception as e:
            print(f"Failed to convert {col}: {e}")
    
    return df_copy

def calculate_stay_duration(df, from_col='STAY_FROM_DT', thru_col='STAY_THRU_DT', new_col='STAY_DURATION'):
    """
    Calculate the duration of a patient's stay in days
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing date columns
    from_col : str, default='STAY_FROM_DT'
        Column name for the start date
    thru_col : str, default='STAY_THRU_DT'
        Column name for the end date
    new_col : str, default='STAY_DURATION'
        New column name for the duration
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with the new duration column
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Convert to datetime if not already done
    for col in [from_col, thru_col]:
        if not pd.api.types.is_datetime64_dtype(df_copy[col]):
            df_copy[col] = pd.to_datetime(df_copy[col])
    
    # Calculate duration in days
    df_copy[new_col] = (df_copy[thru_col] - df_copy[from_col]).dt.days
    
    return df_copy

def count_diagnoses(df, prefix='DGNSCD', exclude_empty=True):
    """
    Count the number of diagnoses per patient
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing diagnosis columns
    prefix : str, default='DGNSCD'
        Prefix of diagnosis columns
    exclude_empty : bool, default=True
        Whether to exclude empty/NaN diagnoses
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with a new column for diagnosis count
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Find all diagnosis columns
    dgns_cols = [col for col in df_copy.columns if prefix in col]
    
    # Count non-empty diagnoses
    if exclude_empty:
        df_copy['DGNS_COUNT'] = df_copy[dgns_cols].notna().sum(axis=1)
    else:
        df_copy['DGNS_COUNT'] = len(dgns_cols)
    
    return df_copy

def save_processed_data(df, output_path='processed_train.csv'):
    """
    Save the processed DataFrame to a CSV file
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    output_path : str, default='processed_train.csv'
        Path to save the processed data
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Data saved successfully to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def plot_readmission_distribution(df, target_col='Readmitted_30'):
    """
    Plot the distribution of the readmission target variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the target column
    target_col : str, default='Readmitted_30'
        Column name of the target variable
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df)
    plt.title('Distribution of 30-Day Readmission')
    plt.xlabel('Readmitted within 30 days')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()

if __name__ == "__main__":
    # Example usage
    df = load_train_data()
    if df is not None:
        print("\nSample data:")
        print(df.head())
        
        # Get basic information about the data
        get_data_info(df)
        
        # Process the data
        df = convert_dates(df)
        df = calculate_stay_duration(df)
        df = count_diagnoses(df)
        
        # Save the processed data
        save_processed_data(df) 