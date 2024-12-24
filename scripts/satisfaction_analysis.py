import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from load_data import load_data_from_postgres
from overview_analysis import clean_data
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

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
    - Sessions frequency
    - Total session duration
    - Total data volume
    - Average TCP retransmission
    - Average RTT
    - Handset type
    - Average throughput
    """
    user_aggregated_data = df.groupby('MSISDN/Number').agg(
        sessions_frequency=('Bearer Id', 'count'),
        total_session_duration=('Dur. (ms)', 'sum'),
        total_download_data=('Total DL (Bytes)', 'sum'),
        total_upload_data=('Total UL (Bytes)', 'sum'),
        avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
        avg_rtt=('Avg RTT DL (ms)', 'mean'),
        handset_type=('Handset Type', 'first'),
        avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')
    ).reset_index()
    
    user_aggregated_data['total_data_volume'] = user_aggregated_data['total_download_data'] + user_aggregated_data['total_upload_data']
    
    return user_aggregated_data

def calculate_engagement_score(user_aggregated_data, kmeans):
    """
    Calculate the engagement score for each user.
    """
    less_engaged_cluster_center = kmeans.cluster_centers_[0]
    user_aggregated_data['engagement_score'] = euclidean_distances(user_aggregated_data[['sessions_frequency', 'total_session_duration', 'total_data_volume']], [less_engaged_cluster_center]).flatten()
    return user_aggregated_data

def calculate_experience_score(user_aggregated_data, kmeans):
    """
    Calculate the experience score for each user.
    """
    worst_experience_cluster_center = kmeans.cluster_centers_[0]
    user_aggregated_data['experience_score'] = euclidean_distances(user_aggregated_data[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']], [worst_experience_cluster_center]).flatten()
    return user_aggregated_data

def calculate_satisfaction_score(user_aggregated_data):
    """
    Calculate the satisfaction score for each user.
    """
    user_aggregated_data['satisfaction_score'] = (user_aggregated_data['engagement_score'] + user_aggregated_data['experience_score']) / 2
    return user_aggregated_data

def build_regression_model(user_aggregated_data):
    """
    Build a regression model to predict the satisfaction score of a customer.
    """
    X = user_aggregated_data[['engagement_score', 'experience_score']]
    y = user_aggregated_data['satisfaction_score']
    model = LinearRegression()
    model.fit(X, y)
    return model

def run_kmeans(user_aggregated_data, n_clusters=2):
    """
    Run a k-means clustering on the engagement and experience scores.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    user_aggregated_data['cluster'] = kmeans.fit_predict(user_aggregated_data[['engagement_score', 'experience_score']])
    return user_aggregated_data, kmeans

def aggregate_scores_per_cluster(user_aggregated_data):
    """
    Aggregate the average satisfaction and experience score per cluster.
    """
    cluster_stats = user_aggregated_data.groupby('cluster').agg({
        'engagement_score': 'mean',
        'experience_score': 'mean',
        'satisfaction_score': 'mean'
    }).reset_index()
    return cluster_stats

def save_to_csv(user_aggregated_data, directory='../Models'):
    """
    Save the final table containing all user ID + engagement, experience & satisfaction scores to a CSV file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    user_aggregated_data.to_csv(os.path.join(directory, 'user_satisfaction.csv'), index=False)