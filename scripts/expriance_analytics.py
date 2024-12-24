import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from load_data import load_data_from_postgres
from overview_analysis import clean_data

def load_and_prepare_data(query):
    """
    Load data from PostgreSQL and prepare it by filling missing values.
    """
    df = load_data_from_postgres(query)
    df = clean_data(df)
    return df

def treat_missing_and_outliers(df):
    """
    Treat missing values and outliers by replacing with the mean or the mode of the corresponding variable.
    """
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].mean())
        df[column] = np.where(df[column] > df[column].mean() + 3 * df[column].std(), df[column].mean(), df[column])
        df[column] = np.where(df[column] < df[column].mean() - 3 * df[column].std(), df[column].mean(), df[column])
    
    for column in df.select_dtypes(include=[object]).columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    
    return df

def aggregate_per_customer(df):
    """
    Aggregate, per customer, the following information:
    - Average TCP retransmission
    - Average RTT
    - Handset type
    - Average throughput
    """
    user_aggregated_data = df.groupby('MSISDN/Number').agg(
        avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
        avg_rtt=('Avg RTT DL (ms)', 'mean'),
        handset_type=('Handset Type', 'first'),
        avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')
    ).reset_index()
    
    return user_aggregated_data

def compute_top_bottom_frequent(df, column):
    """
    Compute & list 10 of the top, bottom, and most frequent values in the dataset for a given column.
    """
    top_10 = df.nlargest(10, column)
    bottom_10 = df.nsmallest(10, column)
    most_frequent_10 = df[column].value_counts().head(10)
    
    return top_10, bottom_10, most_frequent_10

def distribution_per_handset_type(df, column):
    """
    Compute the distribution of the average throughput per handset type and provide interpretation.
    """
    distribution = df.groupby('handset_type')[column].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='handset_type', y=column, data=distribution)
    plt.title(f'Distribution of {column} per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel(f'Average {column}')
    plt.xticks(rotation=90)
    plt.show()
    
    return distribution

def kmeans_clustering(df, n_clusters=3):
    """
    Perform a k-means clustering to segment users into groups of experiences.
    """
    scaler = StandardScaler()
    metrics = ['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']
    df[metrics] = scaler.fit_transform(df[metrics])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[metrics])
    
    return df, kmeans