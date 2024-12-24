import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from load_data import load_data_from_postgres  # Function to load data from PostgreSQL
from overview_analysis import clean_data  # Function to clean data (assumed to be imported)
from dotenv import load_dotenv  # To load environment variables from .env file
import os

# Load environment variables from .env file to access the database connection details
load_dotenv()

# Database credentials from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def load_and_prepare_data(query):
    """
    Load data from PostgreSQL and prepare it by cleaning and filling missing values.
    """
    # Load the data using a custom function that connects to PostgreSQL
    df = load_data_from_postgres(query)
    
    # Clean the data (assumes 'clean_data' function handles missing values and basic cleaning)
    df = clean_data(df)
    
    # Return the cleaned DataFrame
    return df

def treat_missing_and_outliers(df):
    """
    Treat missing values and outliers by replacing them with the mean or the mode of the corresponding variable.
    """
    # Iterate over numerical columns to handle missing values and outliers
    for column in df.select_dtypes(include=[np.number]).columns:
        # Fill missing numerical values with the column's mean
        df[column] = df[column].fillna(df[column].mean())
        
        # Replace outliers (values > mean + 3 * std) with the mean
        df[column] = np.where(df[column] > df[column].mean() + 3 * df[column].std(), df[column].mean(), df[column])
        
        # Replace outliers (values < mean - 3 * std) with the mean
        df[column] = np.where(df[column] < df[column].mean() - 3 * df[column].std(), df[column].mean(), df[column])
    
    # Iterate over categorical columns to handle missing values
    for column in df.select_dtypes(include=[object]).columns:
        # Fill missing categorical values with the mode (most frequent value)
        df[column] = df[column].fillna(df[column].mode()[0])
    
    # Return the cleaned DataFrame with treated missing values and outliers
    return df

def aggregate_per_customer(df):
    """
    Aggregate, per customer (MSISDN/Number), the following information:
    - Sessions frequency
    - Total session duration
    - Total data volume
    - Average TCP retransmission
    - Average RTT
    - Handset type
    - Average throughput
    """
    # Group the data by 'MSISDN/Number' and calculate relevant metrics for each user
    user_aggregated_data = df.groupby('MSISDN/Number').agg(
        sessions_frequency=('Bearer Id', 'count'),  # Count the number of sessions (Bearer Id occurrences)
        total_session_duration=('Dur. (ms)', 'sum'),  # Sum of session durations in ms
        total_download_data=('Total DL (Bytes)', 'sum'),  # Total download data in Bytes
        total_upload_data=('Total UL (Bytes)', 'sum'),  # Total upload data in Bytes
        avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),  # Average TCP retransmission volume
        avg_rtt=('Avg RTT DL (ms)', 'mean'),  # Average Round-Trip Time (RTT) in ms
        handset_type=('Handset Type', 'first'),  # Most frequent handset type for the user
        avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')  # Average throughput in kbps
    ).reset_index()
    
    # Calculate total data volume as the sum of download and upload data
    user_aggregated_data['total_data_volume'] = user_aggregated_data['total_download_data'] + user_aggregated_data['total_upload_data']
    
    # Return the aggregated DataFrame
    return user_aggregated_data

def calculate_engagement_score(user_aggregated_data, kmeans):
    """
    Calculate the engagement score for each user based on their session-related data.
    """
    # Use the cluster center representing less engaged users for comparison
    less_engaged_cluster_center = kmeans.cluster_centers_[0]
    
    # Calculate the Euclidean distance between each user's engagement features and the 'less engaged' cluster center
    user_aggregated_data['engagement_score'] = euclidean_distances(
        user_aggregated_data[['sessions_frequency', 'total_session_duration', 'total_data_volume']], 
        [less_engaged_cluster_center]
    ).flatten()  # Flatten to convert the result from a 2D array to a 1D array
    
    # Return the DataFrame with the engagement score added
    return user_aggregated_data

def calculate_experience_score(user_aggregated_data, kmeans):
    """
    Calculate the experience score for each user based on their experience-related data.
    """
    # Use the cluster center representing the worst experience for comparison
    worst_experience_cluster_center = kmeans.cluster_centers_[0]
    
    # Calculate the Euclidean distance between each user's experience features and the 'worst experience' cluster center
    user_aggregated_data['experience_score'] = euclidean_distances(
        user_aggregated_data[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']], 
        [worst_experience_cluster_center]
    ).flatten()  # Flatten to convert the result from a 2D array to a 1D array
    
    # Return the DataFrame with the experience score added
    return user_aggregated_data

def calculate_satisfaction_score(user_aggregated_data):
    """
    Calculate the satisfaction score for each user by averaging engagement and experience scores.
    """
    # Calculate satisfaction score as the mean of engagement and experience scores
    user_aggregated_data['satisfaction_score'] = (user_aggregated_data['engagement_score'] + user_aggregated_data['experience_score']) / 2
    
    # Return the DataFrame with the satisfaction score added
    return user_aggregated_data

def build_regression_model(user_aggregated_data):
    """
    Build a linear regression model to predict the satisfaction score of a user based on engagement and experience scores.
    """
    # Define the features (engagement and experience scores) and target (satisfaction score)
    X = user_aggregated_data[['engagement_score', 'experience_score']]
    y = user_aggregated_data['satisfaction_score']
    
    # Initialize the linear regression model
    model = LinearRegression()
    
    # Train the model using the features and target
    model.fit(X, y)
    
    # Return the trained model
    return model

def run_kmeans(user_aggregated_data, n_clusters=2):
    """
    Run K-Means clustering on the engagement and experience scores to segment users.
    """
    # Initialize KMeans clustering with the specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Apply clustering to the engagement and experience scores
    user_aggregated_data['cluster'] = kmeans.fit_predict(user_aggregated_data[['engagement_score', 'experience_score']])
    
    # Return the DataFrame with cluster labels and the trained KMeans model
    return user_aggregated_data, kmeans

def aggregate_scores_per_cluster(user_aggregated_data):
    """
    Aggregate and calculate the average engagement, experience, and satisfaction scores per cluster.
    """
    # Group by cluster and calculate mean scores for each cluster
    cluster_stats = user_aggregated_data.groupby('cluster').agg({
        'engagement_score': 'mean',
        'experience_score': 'mean',
        'satisfaction_score': 'mean'
    }).reset_index()
    
    # Return the aggregated cluster statistics
    return cluster_stats

def save_to_csv(user_aggregated_data, directory='../Models'):
    """
    Save the final table containing all user IDs + engagement, experience & satisfaction scores to a CSV file.
    """
    # Check if the directory exists, and create it if not
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the DataFrame to a CSV file in the specified directory
    user_aggregated_data.to_csv(os.path.join(directory, 'user_satisfaction.csv'), index=False)
