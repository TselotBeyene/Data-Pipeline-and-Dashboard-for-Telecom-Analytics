import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from load_data import load_data_from_postgres  # Custom function to load data from PostgreSQL
from overview_analysis import clean_data  # Custom function to clean data

def load_and_prepare_data(query):
    """
    Load data from PostgreSQL and prepare it by filling missing values.
    
    Parameters:
    query (str): The SQL query used to retrieve data from the database.
    
    Returns:
    pd.DataFrame: The cleaned and prepared dataframe.
    """
    # Load data using a custom function from PostgreSQL
    df = load_data_from_postgres(query)
    
    # Clean the data using a custom cleaning function
    df = clean_data(df)
    
    return df

def treat_missing_and_outliers(df):
    """
    Treat missing values and outliers by replacing with the mean or the mode of the corresponding variable.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to treat missing values and outliers.
    
    Returns:
    pd.DataFrame: The dataframe after missing values and outliers are handled.
    """
    # Iterate through numeric columns
    for column in df.select_dtypes(include=[np.number]).columns:
        # Fill missing values with the mean of the column
        df[column] = df[column].fillna(df[column].mean())
        
        # Replace outliers (values > 3 standard deviations from the mean) with the mean
        df[column] = np.where(df[column] > df[column].mean() + 3 * df[column].std(), df[column].mean(), df[column])
        df[column] = np.where(df[column] < df[column].mean() - 3 * df[column].std(), df[column].mean(), df[column])
    
    # Iterate through categorical (object) columns
    for column in df.select_dtypes(include=[object]).columns:
        # Fill missing values with the mode (most frequent value) of the column
        df[column] = df[column].fillna(df[column].mode()[0])
    
    return df

def aggregate_per_customer(df):
    """
    Aggregate, per customer, the following information:
    - Average TCP retransmission
    - Average RTT
    - Handset type
    - Average throughput
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing customer data.
    
    Returns:
    pd.DataFrame: A dataframe with aggregated customer data.
    """
    # Group the data by 'MSISDN/Number' (which represents the customer) and calculate aggregates
    user_aggregated_data = df.groupby('MSISDN/Number').agg(
        avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),  # Calculate mean TCP retransmission
        avg_rtt=('Avg RTT DL (ms)', 'mean'),  # Calculate mean RTT
        handset_type=('Handset Type', 'first'),  # Take the first value of 'Handset Type' for the customer
        avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')  # Calculate mean throughput
    ).reset_index()  # Reset index to make 'MSISDN/Number' a column again
    
    return user_aggregated_data

def compute_top_bottom_frequent(df, column):
    """
    Compute & list 10 of the top, bottom, and most frequent values in the dataset for a given column.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    column (str): The column name to perform the analysis on.
    
    Returns:
    tuple: Three DataFrames (top 10, bottom 10, and most frequent 10 values).
    """
    # Get the top 10 values based on the column
    top_10 = df.nlargest(10, column)
    
    # Get the bottom 10 values based on the column
    bottom_10 = df.nsmallest(10, column)
    
    # Get the 10 most frequent values in the column
    most_frequent_10 = df[column].value_counts().head(10)
    
    return top_10, bottom_10, most_frequent_10

def distribution_per_handset_type(df, column):
    """
    Compute the distribution of the average throughput per handset type and provide interpretation.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    column (str): The column (e.g., 'Avg Bearer TP DL (kbps)') to compute the distribution for.
    
    Returns:
    pd.DataFrame: A dataframe with the distribution of the specified column per handset type.
    """
    # Group the data by 'handset_type' and calculate the mean of the specified column
    distribution = df.groupby('handset_type')[column].mean().reset_index()
    
    # Create a bar plot to visualize the distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x='handset_type', y=column, data=distribution)
    plt.title(f'Distribution of {column} per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel(f'Average {column}')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.show()
    
    return distribution

def kmeans_clustering(df, n_clusters=3):
    """
    Perform k-means clustering to segment users into groups based on their experience.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing the user data.
    n_clusters (int): The number of clusters to form (default is 3).
    
    Returns:
    tuple: A tuple containing the dataframe with cluster labels and the trained k-means model.
    """
    # Standardize the data (scaling the metrics to have zero mean and unit variance)
    scaler = StandardScaler()
    metrics = ['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']
    df[metrics] = scaler.fit_transform(df[metrics])
    
    # Initialize the KMeans model and fit it to the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[metrics])  # Assign cluster labels to each user
    
    return df, kmeans
