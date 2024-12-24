import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from load_data import load_data_from_postgres  # Custom function to load data from PostgreSQL
from overview_analysis import clean_data  # Custom function to clean data

def load_and_prepare_data(query):
    """
    Load data from PostgreSQL and prepare it by cleaning and filling missing values.
    
    Parameters:
    query (str): The SQL query used to retrieve data from the database.
    
    Returns:
    pd.DataFrame: The cleaned and prepared dataframe.
    """
    # Load data from PostgreSQL using a custom function
    df = load_data_from_postgres(query)
    
    # Clean the data using a custom function (e.g., handling missing values, outliers, etc.)
    df = clean_data(df)
    
    return df

def describe_variables(df):
    """
    Describe all relevant variables and their associated data types.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to describe.
    
    Returns:
    tuple: A tuple containing the description of variables and their data types.
    """
    # Generate a summary description of all variables (including categorical ones)
    description = df.describe(include='all')
    
    # Get data types for each column
    data_types = df.dtypes
    
    return description, data_types

def segment_users(df):
    """
    Segment users into decile classes based on total session duration and compute the total data usage (DL+UL) per decile class.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing session data.
    
    Returns:
    pd.DataFrame: A dataframe with the aggregated data for each decile class.
    """
    # Calculate the total session duration and total data usage (DL+UL)
    df['total_duration'] = df['Dur. (ms)']
    df['total_data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    
    # Segment the users into 10 deciles based on the total session duration
    df['decile'] = pd.qcut(df['total_duration'], 10, labels=False, duplicates='drop')
    
    # Aggregate data by decile: total session duration and total data usage
    decile_data = df.groupby('decile').agg(
        total_duration=('total_duration', 'sum'),
        total_data=('total_data', 'sum')
    ).reset_index()  # Reset index to turn 'decile' into a column
    
    return decile_data

def basic_metrics(df):
    """
    Analyze basic metrics (mean, median, etc.) for each numeric variable in the dataset.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    
    Returns:
    pd.DataFrame: A dataframe containing basic descriptive statistics (mean, median, etc.) for each column.
    """
    # Generate basic descriptive statistics for the dataframe
    metrics = df.describe()
    
    return metrics

def non_graphical_univariate_analysis(df):
    """
    Conduct a non-graphical univariate analysis by computing dispersion parameters for each quantitative variable.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    
    Returns:
    pd.DataFrame: A dataframe containing dispersion parameters (mean, std, min, 25%, 50%, 75%, max) for each column.
    """
    # Get the dispersion parameters for each numerical column
    dispersion_params = df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    return dispersion_params

def graphical_univariate_analysis(df):
    """
    Conduct a graphical univariate analysis by visualizing the distribution of each numeric variable.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    
    Displays:
    Histogram and KDE plots for each numeric column in the dataframe.
    """
    # Iterate through all numeric columns in the dataframe
    for column in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(10, 6))
        
        # Create a histogram with KDE (Kernel Density Estimation) to visualize the distribution of the column
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

def bivariate_analysis(df):
    """
    Explore the relationship between each application and the total data (DL+UL) using scatter plots.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing session data.
    
    Displays:
    Scatter plots showing the relationship between each application data usage and the total data (DL+UL).
    """
    # Calculate the total data (DL + UL)
    df['total_data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    
    # List of applications to analyze (data usage per application)
    applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                    'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    # For each application, create a scatter plot showing the relationship between the app and the total data
    for app in applications:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[app], y=df['total_data'])
        plt.title(f'Relationship between {app} and Total Data')
        plt.xlabel(app)
        plt.ylabel('Total Data (DL + UL)')
        plt.show()

def correlation_analysis(df):
    """
    Compute a correlation matrix for the specified variables and visualize it.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    
    Returns:
    pd.DataFrame: A dataframe representing the correlation matrix for the specified variables.
    
    Displays:
    A heatmap showing the correlation matrix of the selected variables.
    """
    # List of variables to compute correlation
    variables = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    # Compute the correlation matrix for the selected variables
    correlation_matrix = df[variables].corr()
    
    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  # Annotate values on the heatmap
    plt.title('Correlation Matrix')
    plt.show()
    
    return correlation_matrix

def dimensionality_reduction(df):
    """
    Perform a Principal Component Analysis (PCA) to reduce the dimensions of the data.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing the application data.
    
    Returns:
    tuple: A tuple containing the PCA results (reduced data) and the explained variance ratio.
    
    Displays:
    A scatter plot showing the result of the PCA transformation.
    """
    # List of application data columns to apply PCA on
    variables = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    # Initialize PCA with 2 components (reduce the data to 2D)
    pca = PCA(n_components=2)
    
    # Perform PCA and reduce the dimensions of the selected data
    pca_result = pca.fit_transform(df[variables].fillna(0))  # Fill missing values with 0
    
    # Create a scatter plot to visualize the result of the PCA transformation
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    
    # Return the PCA result and the explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance
