import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from load_data import load_data_from_postgres
from overview_analysis import clean_data

def load_and_prepare_data(query):
    """
    Load data from PostgreSQL and prepare it by filling missing values.
    """
    df = load_data_from_postgres(query)
    df = clean_data(df)
    return df

def describe_variables(df):
    """
    Describe all relevant variables and associated data types.
    """
    description = df.describe(include='all')
    data_types = df.dtypes
    return description, data_types

def segment_users(df):
    """
    Segment the users into the top five decile classes based on the total duration for all sessions
    and compute the total data (DL+UL) per decile class.
    """
    df['total_duration'] = df['Dur. (ms)']
    df['total_data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    df['decile'] = pd.qcut(df['total_duration'], 10, labels=False, duplicates='drop')
    
    decile_data = df.groupby('decile').agg(
        total_duration=('total_duration', 'sum'),
        total_data=('total_data', 'sum')
    ).reset_index()
    
    return decile_data

def basic_metrics(df):
    """
    Analyze the basic metrics (mean, median, etc) in the Dataset.
    """
    metrics = df.describe()
    return metrics

def non_graphical_univariate_analysis(df):
    """
    Conduct a Non-Graphical Univariate Analysis by computing dispersion parameters for each quantitative variable.
    """
    dispersion_params = df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    return dispersion_params

def graphical_univariate_analysis(df):
    """
    Conduct a Graphical Univariate Analysis by identifying the most suitable plotting options for each variable.
    """
    for column in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

def bivariate_analysis(df):
    """
    Explore the relationship between each application & the total DL+UL data using appropriate methods.
    """
    df['total_data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                    'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    for app in applications:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[app], y=df['total_data'])
        plt.title(f'Relationship between {app} and Total Data')
        plt.xlabel(app)
        plt.ylabel('Total Data (DL + UL)')
        plt.show()

def correlation_analysis(df):
    """
    Compute a correlation matrix for the specified variables and interpret your findings.
    """
    variables = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    correlation_matrix = df[variables].corr()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
    return correlation_matrix

def dimensionality_reduction(df):
    """
    Perform a principal component analysis to reduce the dimensions of your data.
    """
    variables = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[variables].fillna(0))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance