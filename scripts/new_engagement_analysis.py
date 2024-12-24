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

def aggregate_metrics(df):
    """
    Aggregate the engagement metrics per customer id (MSISDN).
    """
    user_aggregated_data = df.groupby('MSISDN/Number').agg(
        sessions_frequency=('Bearer Id', 'count'),
        total_session_duration=('Dur. (ms)', 'sum'),
        total_download_data=('Total DL (Bytes)', 'sum'),
        total_upload_data=('Total UL (Bytes)', 'sum')
    ).reset_index()
    
    user_aggregated_data['total_data_volume'] = user_aggregated_data['total_download_data'] + user_aggregated_data['total_upload_data']
    
    return user_aggregated_data

def top_10_customers(user_aggregated_data):
    """
    Report the top 10 customers per engagement metric.
    """
    top_10_sessions = user_aggregated_data.nlargest(10, 'sessions_frequency')
    top_10_duration = user_aggregated_data.nlargest(10, 'total_session_duration')
    top_10_data_volume = user_aggregated_data.nlargest(10, 'total_data_volume')
    
    return top_10_sessions, top_10_duration, top_10_data_volume

def normalize_and_cluster(user_aggregated_data):
    """
    Normalize each engagement metric and run a k-means (k=3) to classify customers in three groups of engagement.
    """
    scaler = StandardScaler()
    metrics = ['sessions_frequency', 'total_session_duration', 'total_download_data', 'total_upload_data', 'total_data_volume']
    user_aggregated_data[metrics] = scaler.fit_transform(user_aggregated_data[metrics])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_aggregated_data['cluster'] = kmeans.fit_predict(user_aggregated_data[metrics])
    
    return user_aggregated_data, kmeans

def cluster_statistics(user_aggregated_data):
    """
    Compute the minimum, maximum, average & total non-normalized metrics for each cluster.
    """
    metrics = ['sessions_frequency', 'total_session_duration', 'total_download_data', 'total_upload_data', 'total_data_volume']
    cluster_stats = user_aggregated_data.groupby('cluster')[metrics].agg(['min', 'max', 'mean', 'sum']).reset_index()
    
    return cluster_stats

def top_10_users_per_application(df):
    """
    Aggregate user total traffic per application and derive the top 10 most engaged users per application.
    """
    applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                    'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    top_10_users = {}
    for app in applications:
        app_data = df.groupby('MSISDN/Number')[app].sum().reset_index()
        top_10_users[app] = app_data.nlargest(10, app)
    
    return top_10_users

def plot_top_3_applications(top_10_users):
    """
    Plot the top 3 most used applications using appropriate charts.
    """
    top_3_apps = sorted(top_10_users, key=lambda x: top_10_users[x][x].sum(), reverse=True)[:3]
    
    for app in top_3_apps:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='MSISDN/Number', y=app, data=top_10_users[app])
        plt.title(f'Top 10 Users for {app}')
        plt.xlabel('User ID')
        plt.ylabel('Total Data (Bytes)')
        plt.xticks(rotation=90)
        plt.show()

def elbow_method(user_aggregated_data):
    """
    Determine the optimized value of k using the elbow method.
    """
    scaler = StandardScaler()
    metrics = ['sessions_frequency', 'total_session_duration', 'total_download_data', 'total_upload_data', 'total_data_volume']
    user_aggregated_data[metrics] = scaler.fit_transform(user_aggregated_data[metrics])
    
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(user_aggregated_data[metrics])
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()
