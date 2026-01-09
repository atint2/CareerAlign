import pandas as pd
from datetime import datetime

def load_data(file_path):
    """Load data from a CSV file"""
    return pd.read_csv(file_path)

def convert_to_datetime(date_str):
    """Convert unix timestamp string to datetime object"""
    unix_timestamp = int(date_str) // 1000
    return datetime.fromtimestamp(unix_timestamp)

def exploration_summary(df):
    """Generate a summary of the DataFrame"""
    # Display first few rows
    print("First few rows of the DataFrame:")
    print(df.head())

    # Display info
    print("\nDataFrame Info:")
    print(df.info())

    # Display minimum and maximum values for original listing times
    min_listed_time = df['original_listed_time'].min()
    max_listed_time = df['original_listed_time'].max()
    min_listed_time = convert_to_datetime(min_listed_time)
    max_listed_time = convert_to_datetime(max_listed_time)  
    print(f"\nOriginal Listed Time Range: {min_listed_time} to {max_listed_time}")

def examine_column(df, column_name):
    """Examine a specific column in the DataFrame"""
    print(f"Column: {column_name}")
    print(f"Data Type: {df[column_name].dtype}")
    print(f"Unique Values: {df[column_name].nunique()}")
    print(f"Sample Values: {df[column_name].unique()[:5]}\n")

def main():
    # Load dataset
    df = load_data('data/raw/arshkon-linkedin-dataset.csv')

    # Summarize the DataFrame
    exploration_summary(df)

    # Examine targeted columns
    columns_to_examine = ['title', 'description', 'company_name', 'formatted_work_type', 'formatted_experience_level']
    print("\nExamining targeted columns:")
    for column in columns_to_examine:
        examine_column(df, column)

    # Examine 'formatted_work_type' and 'formatted_experience_level' to understand all unique values
    for column in ['formatted_work_type', 'formatted_experience_level']:
        unique_values = df[column].unique()
        print(f"All unique values in '{column}': {unique_values}")

    # Find average, min, and max number of tokens in job descriptions
    df['description_token_count'] = df['description'].apply(lambda x: len(str(x).split()))
    average_token_count = df['description_token_count'].mean()
    min_token_count = df['description_token_count'].min()
    max_token_count = df['description_token_count'].max()
    std_token_count = df['description_token_count'].std()
    print(f"\nAverage number of tokens in job descriptions: {average_token_count}")
    print(f"Minimum number of tokens in job descriptions: {min_token_count}")
    print(f"Maximum number of tokens in job descriptions: {max_token_count}")
    print(f"Standard deviation of token counts: {std_token_count}")

# Run exploratory data analysis
if __name__ == "__main__":
    main()
