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
    
    This function loads data from a PostgreSQL database using a provided query, 
    cleans the data by handling missing values, and returns the cleaned DataFrame.
    """
    # Load raw data using the query
    df = load_data_from_postgres(query)
    
    # Clean the data by handling missing values, duplicates, etc.
    df = clean_data(df)
    
    # Return the cleaned DataFrame
    return df

def aggregate_metrics(df):
    """
    Aggregate the engagement metrics per customer ID (MSISDN).
    
    This function groups the data by the customer ID and aggregates various 
    metrics like session frequency, total session duration, and data usage (DL and UL).
    """
    user_aggregated_data = df.groupby('MSISDN/Number').agg(
        sessions_frequency=('Bearer Id', 'count'),  # Count sessions per user
        total_session_duration=('Dur. (ms)', 'sum'),  # Sum of session durations per user
        total_download_data=('Total DL (Bytes)', 'sum'),  # Total download data per user
        total_upload_data=('Total UL (Bytes)', 'sum')  # Total upload data per user
    ).reset_index()
    
    # Add a new column for the total data volume (download + upload data)
    user_aggregated_data['total_data_volume'] = user_aggregated_data['total_download_data'] + user_aggregated_data['total_upload_data']
    
    # Return the aggregated data per user
    return user_aggregated_data

def top_10_customers(user_aggregated_data):
    """
    Report the top 10 customers per engagement metric.
    
    This function returns the top 10 customers based on the frequency of sessions,
    the total session duration, and the total data volume.
    """
    # Get the top 10 users based on session frequency
    top_10_sessions = user_aggregated_data.nlargest(10, 'sessions_frequency')
    
    # Get the top 10 users based on total session duration
    top_10_duration = user_aggregated_data.nlargest(10, 'total_session_duration')
    
    # Get the top 10 users based on total data volume
    top_10_data_volume = user_aggregated_data.nlargest(10, 'total_data_volume')
    
    # Return all the top 10 results
    return top_10_sessions, top_10_duration, top_10_data_volume

def normalize_and_cluster(user_aggregated_data):
    """
    Normalize each engagement metric and run a k-means (k=3) to classify customers into three groups of engagement.
    
    This function normalizes the engagement metrics using StandardScaler and then performs 
    k-means clustering to classify customers into three groups based on their engagement metrics.
    """
    # Normalize the metrics using StandardScaler
    scaler = StandardScaler()
    metrics = ['sessions_frequency', 'total_session_duration', 'total_download_data', 'total_upload_data', 'total_data_volume']
    user_aggregated_data[metrics] = scaler.fit_transform(user_aggregated_data[metrics])
    
    # Perform k-means clustering (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_aggregated_data['cluster'] = kmeans.fit_predict(user_aggregated_data[metrics])
    
    # Return the clustered data and k-means model
    return user_aggregated_data, kmeans

def cluster_statistics(user_aggregated_data):
    """
    Compute the minimum, maximum, average, and total non-normalized metrics for each cluster.
    
    This function computes summary statistics (min, max, mean, sum) for each of the engagement metrics 
    within each cluster created by the k-means algorithm.
    """
    metrics = ['sessions_frequency', 'total_session_duration', 'total_download_data', 'total_upload_data', 'total_data_volume']
    
    # Group by cluster and calculate the stats for each metric
    cluster_stats = user_aggregated_data.groupby('cluster')[metrics].agg(['min', 'max', 'mean', 'sum']).reset_index()
    
    # Return the computed statistics per cluster
    return cluster_stats

def top_10_users_per_application(df):
    """
    Aggregate user total traffic per application and derive the top 10 most engaged users per application.
    
    This function computes the total traffic for each user per application and returns the top 10 users for 
    each application based on their total data usage.
    """
    applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                    'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    # Dictionary to store the top 10 users per application
    top_10_users = {}
    
    # Loop through each application and find the top 10 users
    for app in applications:
        app_data = df.groupby('MSISDN/Number')[app].sum().reset_index()
        top_10_users[app] = app_data.nlargest(10, app)
    
    # Return the top 10 users per application
    return top_10_users

def plot_top_3_applications(top_10_users):
    """
    Plot the top 3 most used applications using appropriate charts.
    
    This function identifies the top 3 most popular applications based on total user traffic 
    and generates bar plots showing the top 10 users for each of these applications.
    """
    # Sort applications by total data usage and select the top 3
    top_3_apps = sorted(top_10_users, key=lambda x: top_10_users[x][x].sum(), reverse=True)[:3]
    
    # Plot bar charts for the top 3 applications
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
    
    This function runs the k-means algorithm for different values of k (from 1 to 10) and 
    plots the Sum of Squared Errors (SSE) to identify the optimal number of clusters.
    """
    # Normalize the metrics using StandardScaler
    scaler = StandardScaler()
    metrics = ['sessions_frequency', 'total_session_duration', 'total_download_data', 'total_upload_data', 'total_data_volume']
    user_aggregated_data[metrics] = scaler.fit_transform(user_aggregated_data[metrics])
    
    # List to store SSE values for each k
    sse = []
    
    # Loop through k values from 1 to 10
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(user_aggregated_data[metrics])
        sse.append(kmeans.inertia_)  # Inertia is the SSE (Sum of Squared Errors)
    
    # Plot the SSE values for different k values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()
