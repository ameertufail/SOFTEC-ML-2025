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

def handle_missing_values(df, threshold=0.4, numerical_strategy='mean', categorical_strategy='mode', exclude_columns=None):
    """
    Handle missing values in the DataFrame:
    - Drop columns with missing values >= threshold (default 40%)
    - For remaining columns:
      - Replace null values in numerical columns with mean or median
      - Replace null values in categorical columns with mode (most frequent value)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing missing values
    threshold : float, default=0.4
        Threshold for dropping columns (0.4 = 40% missing values)
    numerical_strategy : str, default='mean'
        Strategy for handling numerical nulls ('mean' or 'median')
    categorical_strategy : str, default='mode'
        Strategy for handling categorical nulls (only 'mode' is supported)
    exclude_columns : list, default=None
        List of column names to exclude from imputation and dropping
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with dropped columns and imputed values
    """
    if exclude_columns is None:
        exclude_columns = []
        
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Calculate percentage of missing values for each column
    missing_percentage = df_copy.isnull().mean()
    
    # Identify columns to drop (>= threshold missing values)
    cols_to_drop = [col for col in missing_percentage[missing_percentage >= threshold].index 
                    if col not in exclude_columns]
    
    # Drop columns with missing values >= threshold
    if cols_to_drop:
        print(f"Dropping columns with >= {threshold*100}% missing values:")
        for col in cols_to_drop:
            print(f"  - {col}: {missing_percentage[col]*100:.2f}% missing")
        df_copy = df_copy.drop(columns=cols_to_drop)
    else:
        print(f"No columns with >= {threshold*100}% missing values")
    
    # Get numeric and categorical columns after dropping
    numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out excluded columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
    categorical_cols = [col for col in categorical_cols if col not in exclude_columns]
    
    # Display missing value counts before imputation
    print("\nMissing values before imputation:")
    print(df_copy[numeric_cols + categorical_cols].isnull().sum())
    
    # Handle missing values in numeric columns
    for col in numeric_cols:
        if df_copy[col].isnull().sum() > 0:
            if numerical_strategy == 'mean':
                fill_value = df_copy[col].mean()
                df_copy[col].fillna(fill_value, inplace=True)
                print(f"Filled nulls in {col} with mean: {fill_value:.2f}")
            elif numerical_strategy == 'median':
                fill_value = df_copy[col].median()
                df_copy[col].fillna(fill_value, inplace=True)
                print(f"Filled nulls in {col} with median: {fill_value:.2f}")
            else:
                print(f"Invalid numerical strategy '{numerical_strategy}', column {col} not imputed")
    
    # Handle missing values in categorical columns
    for col in categorical_cols:
        if df_copy[col].isnull().sum() > 0:
            if categorical_strategy == 'mode':
                # Get the most frequent value (mode)
                mode_value = df_copy[col].mode()[0]
                df_copy[col].fillna(mode_value, inplace=True)
                print(f"Filled nulls in {col} with mode: {mode_value}")
            else:
                print(f"Invalid categorical strategy '{categorical_strategy}', column {col} not imputed")
    
    # Display missing value counts after imputation
    print("\nMissing values after imputation:")
    print(df_copy[numeric_cols + categorical_cols].isnull().sum())
    
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

def display_missing_percentages(df):
    """
    Display the percentage of missing values for each column in the DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    
    Returns:
    --------
    pd.Series
        Series containing the percentage of missing values for each column
    """
    # Calculate missing value percentages
    missing_percentage = df.isnull().mean() * 100
    
    # Sort from highest to lowest percentage
    missing_percentage = missing_percentage.sort_values(ascending=False)
    
    # Format to show 2 decimal places and add % sign
    formatted_percentages = pd.DataFrame({
        'Column': missing_percentage.index,
        'Missing (%)': missing_percentage.values
    })
    
    # Display only columns with missing values
    non_zero_percentages = formatted_percentages[formatted_percentages['Missing (%)'] > 0]
    
    if len(non_zero_percentages) > 0:
        print("\nColumns with missing values (sorted by percentage):")
        for _, row in non_zero_percentages.iterrows():
            print(f"{row['Column']}: {row['Missing (%)']:.2f}%")
        
        # Create a bar plot of missing percentages
        plt.figure(figsize=(10, 6))
        plt.bar(non_zero_percentages['Column'], non_zero_percentages['Missing (%)'])
        plt.xticks(rotation=90)
        plt.title('Percentage of Missing Values by Column')
        plt.xlabel('Column')
        plt.ylabel('Missing (%)')
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo missing values found in the dataset.")
    
    return missing_percentage

def analyze_column_relationship(df, col1='DGNSCD02', col2='PRCDRCD02'):
    """
    Analyze the relationship between two columns in the DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the columns
    col1 : str, default='DGNSCD02'
        First column to analyze
    col2 : str, default='PRCDRCD02'
        Second column to analyze
    """
    
    # Check if columns exist in DataFrame
    if col1 not in df.columns or col2 not in df.columns:
        print(f"One or both columns ({col1}, {col2}) not found in DataFrame")
        return
    
    # Display basic information about the columns
    print(f"\nAnalyzing relationship between {col1} and {col2}:")
    
    # Calculate missing values in each column
    missing1 = df[col1].isnull().mean() * 100
    missing2 = df[col2].isnull().mean() * 100
    print(f"Missing values - {col1}: {missing1:.2f}%, {col2}: {missing2:.2f}%")
    
    # Count unique values in each column
    unique1 = df[col1].nunique()
    unique2 = df[col2].nunique()
    print(f"Unique values - {col1}: {unique1}, {col2}: {unique2}")
    
    # Check for identical values
    # Filter rows where both columns have non-null values
    common_rows = df.dropna(subset=[col1, col2])
    identical_count = (common_rows[col1] == common_rows[col2]).sum()
    identical_percent = (identical_count / len(common_rows)) * 100 if len(common_rows) > 0 else 0
    
    print(f"Rows with identical values: {identical_count} ({identical_percent:.2f}% of non-null rows)")
    
    # Top matching values
    if identical_count > 0:
        matches = common_rows[common_rows[col1] == common_rows[col2]]
        top_matches = matches[col1].value_counts().head(10)
        print("\nTop 10 matching values:")
        print(top_matches)
    
    # Create a contingency table for the most common values
    top_values1 = df[col1].value_counts().head(10).index
    top_values2 = df[col2].value_counts().head(10).index
    
    # Filter DataFrame to include only rows with top values
    filtered_df = df[df[col1].isin(top_values1) | df[col2].isin(top_values2)]
    
    # Create a crosstab of the top values
    if not filtered_df.empty:
        print("\nContingency table for top values:")
        contingency_table = pd.crosstab(
            filtered_df[col1], 
            filtered_df[col2],
            margins=True,
            dropna=False
        )
        print(contingency_table)
    
    # Visualize relationship if possible
    plt.figure(figsize=(10, 6))
    plt.title(f"Distribution of non-null values in {col1} and {col2}")
    
    # Count non-null values in each column
    col1_counts = df[col1].notna().sum()
    col2_counts = df[col2].notna().sum()
    both_counts = df.dropna(subset=[col1, col2]).shape[0]
    identical_counts = identical_count
    
    # Create a grouped bar chart
    categories = ['Non-null values', 'Identical values']
    col1_data = [col1_counts, 0]  # We don't plot identical values for individual columns
    col2_data = [col2_counts, 0]
    both_data = [both_counts, identical_counts]
    
    x = np.arange(len(categories))
    width = 0.25
    
    plt.bar(x - width, col1_data, width, label=col1)
    plt.bar(x, col2_data, width, label=col2)
    plt.bar(x + width, both_data, width, label='Both columns')
    
    plt.xticks(x, categories)
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, method='pearson', figsize=(20, 16), annot=True, cmap='coolwarm', 
                           mask_upper=True, threshold=None, include_categorical=False):
    """
    Calculate and visualize the correlation matrix for all numeric columns in the DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    method : str, default='pearson'
        Correlation method ('pearson', 'kendall', 'spearman')
    figsize : tuple, default=(20, 16)
        Figure size for the heatmap
    annot : bool, default=True
        Whether to annotate the heatmap with correlation values
    cmap : str, default='coolwarm'
        Colormap for the heatmap
    mask_upper : bool, default=True
        Whether to mask the upper triangle of the correlation matrix
    threshold : float, default=None
        If set, only show correlations above this threshold (absolute value)
    include_categorical : bool, default=False
        Whether to include categorical columns (one-hot encoded)
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Get numeric columns
    numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    
    # Handle categorical columns if requested
    if include_categorical:
        # Find categorical columns
        cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            print(f"One-hot encoding {len(cat_cols)} categorical columns...")
            # One-hot encode categorical columns
            df_encoded = pd.get_dummies(df_copy[cat_cols], drop_first=True)
            # Combine with numeric data
            df_combined = pd.concat([df_copy[numeric_cols], df_encoded], axis=1)
            corr_data = df_combined
            print(f"Correlation matrix will include {corr_data.shape[1]} columns after encoding")
        else:
            corr_data = df_copy[numeric_cols]
    else:
        # Only use numeric columns
        corr_data = df_copy[numeric_cols]
        print(f"Using {len(numeric_cols)} numeric columns for correlation matrix")
    
    # Calculate correlation matrix
    corr_matrix = corr_data.corr(method=method)
    
    # Create mask for upper triangle if requested
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    else:
        mask = np.zeros_like(corr_matrix, dtype=bool)
    
    # Apply threshold if provided
    if threshold is not None:
        # Create a copy of the correlation matrix
        threshold_matrix = corr_matrix.copy()
        # Replace values below threshold with NaN
        threshold_matrix = threshold_matrix.mask(abs(threshold_matrix) < threshold, np.nan)
        # Update the mask to hide values below threshold
        mask = mask | np.isnan(threshold_matrix)
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, 
                annot=annot, 
                cmap=cmap,
                mask=mask,
                vmin=-1, 
                vmax=1, 
                fmt='.2f',
                linewidths=0.5,
                square=True)
    
    plt.title(f'Correlation Matrix ({method})', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Find and print top correlations
    # Flatten the correlation matrix and convert to a Series
    corr_pairs = corr_matrix.unstack()
    # Remove self-correlations (value = 1.0)
    corr_pairs = corr_pairs[corr_pairs < 0.999]
    # Get top positive correlations
    top_positive = corr_pairs.nlargest(20)
    # Get top negative correlations
    top_negative = corr_pairs.nsmallest(20)
    
    print("\nTop Positive Correlations:")
    for idx, val in top_positive.items():
        print(f"{idx[0]} and {idx[1]}: {val:.3f}")
    
    print("\nTop Negative Correlations:")
    for idx, val in top_negative.items():
        print(f"{idx[0]} and {idx[1]}: {val:.3f}")
    
    return corr_matrix

if __name__ == "__main__":
    # Example usage
    df = load_train_data()
    if df is not None:
        print("\nSample data:")
        print(df.head())
        
        # Get basic information about the data
        get_data_info(df)
        
        # Calculate and plot correlation matrix
        corr_matrix = plot_correlation_matrix(df, threshold=0.3)
        
        # Display missing value percentages
        # missing_percentages = display_missing_percentages(df)
        
        # Analyze relationship between DGNSCD02 and PRCDRCD02
        analyze_column_relationship(df, 'DGNSCD02', 'PRCDRCD02')
        
        # Process the data
        df = convert_dates(df)
        df = calculate_stay_duration(df)
        df = count_diagnoses(df)
        
        # Handle missing values
        # df = handle_missing_values(df, threshold=0.4, numerical_strategy='mean', categorical_strategy='mode')
        
        # Save the processed data
        # save_processed_data(df) 