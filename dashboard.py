import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import sys
sys.path.append(os.path.abspath('../scripts'))
# Import utility scripts from the `scripts` folder
from scripts.load_data import load_data_from_postgres
from overview_analysis import describe_variables
from engagement_analysis import aggregate_metrics, normalize_and_cluster
from experience_analytics import aggregate_per_customer, kmeans_clustering
from satisfaction_analysis import calculate_satisfaction_score

# Configure Streamlit
st.set_page_config(page_title="Telecom Insights Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to", ["User Overview Analysis", 
                                  "User Engagement Analysis", 
                                  "Experience Analysis", 
                                  "Satisfaction Analysis"])

# Database query for loading data
QUERY = "SELECT * FROM telecom_data"  # Update this with your actual table or query

# Cache data loading for efficiency
@st.cache
def load_data():
    return load_data_from_postgres(QUERY)

# Load data
df = load_data()

# Define functions for each page
def user_overview_analysis():
    st.title("User Overview Analysis")
    st.subheader("Descriptive Statistics")
    description, data_types = describe_variables(df)
    st.write("### Data Description:")
    st.write(description)
    st.write("### Data Types:")
    st.write(data_types)

    st.subheader("Graphical Univariate Analysis")
    for column in df.select_dtypes(include=[float, int]).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)


def user_engagement_analysis():
    st.title("User Engagement Analysis")
    st.subheader("Engagement Metrics")
    user_aggregated_data = aggregate_metrics(df)
    st.dataframe(user_aggregated_data.head())

    st.subheader("Engagement Clustering")
    user_aggregated_data, kmeans = normalize_and_cluster(user_aggregated_data)
    fig, ax = plt.subplots()
    sns.scatterplot(data=user_aggregated_data, x='sessions_frequency', 
                    y='total_data_volume', hue='cluster', palette='viridis', ax=ax)
    ax.set_title("Engagement Clusters")
    st.pyplot(fig)


def experience_analysis():
    st.title("Experience Analysis")
    st.subheader("Experience Metrics")
    experience_data = aggregate_per_customer(df)
    st.dataframe(experience_data.head())

    st.subheader("Experience Clustering")
    experience_data, kmeans = kmeans_clustering(experience_data, n_clusters=3)
    fig, ax = plt.subplots()
    sns.scatterplot(data=experience_data, x='avg_rtt', y='avg_throughput', 
                    hue='cluster', palette='coolwarm', ax=ax)
    ax.set_title("Experience Clusters")
    st.pyplot(fig)


def satisfaction_analysis():
    st.title("Satisfaction Analysis")
    st.subheader("Satisfaction Scores")
    # Assume engagement and experience scores are precomputed and merged
    user_aggregated_data = aggregate_metrics(df)
    experience_data = aggregate_per_customer(df)

    # Merge engagement and experience scores
    user_aggregated_data['engagement_score'] = user_aggregated_data['sessions_frequency']
    experience_data['experience_score'] = experience_data['avg_rtt']
    satisfaction_data = pd.merge(user_aggregated_data[['MSISDN/Number', 'engagement_score']],
                                 experience_data[['MSISDN/Number', 'experience_score']], 
                                 on='MSISDN/Number')
    satisfaction_data = calculate_satisfaction_score(satisfaction_data)
    st.dataframe(satisfaction_data.head())

    st.subheader("Satisfaction Clustering")
    scaler = StandardScaler()
    satisfaction_metrics = ['engagement_score', 'experience_score']
    satisfaction_data[satisfaction_metrics] = scaler.fit_transform(satisfaction_data[satisfaction_metrics])
    kmeans = KMeans(n_clusters=3, random_state=42)
    satisfaction_data['cluster'] = kmeans.fit_predict(satisfaction_data[satisfaction_metrics])

    fig, ax = plt.subplots()
    sns.scatterplot(data=satisfaction_data, x='engagement_score', y='satisfaction_score', 
                    hue='cluster', palette='plasma', ax=ax)
    ax.set_title("Satisfaction Clusters")
    st.pyplot(fig)


# Page navigation logic
if page == "User Overview Analysis":
    user_overview_analysis()
elif page == "User Engagement Analysis":
    user_engagement_analysis()
elif page == "Experience Analysis":
    experience_analysis()
elif page == "Satisfaction Analysis":
    satisfaction_analysis()
