import pandas as pd
from load_data import load_data_from_postgres
from overview_analysis import clean_data
import os

def load_and_prepare_data(query):
    """
    Load data from PostgreSQL and prepare it by filling missing values.
    """
    df = load_data_from_postgres(query)
    df = clean_data(df)
    return df

def aggregate_user_data(df):
    """
    Aggregate per user the following information:
    - number of xDR sessions
    - Session duration
    - the total download (DL) and upload (UL) data
    - the total data volume (in Bytes) during this session for each application
    """
    user_aggregated_data = df.groupby('MSISDN/Number').agg(
        number_of_xDR_sessions=('Bearer Id', 'count'),
        total_session_duration=('Dur. (ms)', 'sum'),
        total_download_data=('Total DL (Bytes)', 'sum'),
        total_upload_data=('Total UL (Bytes)', 'sum')
    ).reset_index()
    
    user_aggregated_data['total_data_volume'] = user_aggregated_data['total_download_data'] + user_aggregated_data['total_upload_data']
    
    return user_aggregated_data

def save_aggregated_data(user_aggregated_data):
    """
    Save the aggregated user data to a CSV file.
    """
    os.makedirs('data', exist_ok=True)
    user_aggregated_data.to_csv('data/user_aggregated_data.csv', index=False)
