import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data_from_postgres

def load_and_prepare_data(query):
    """
    Load data from PostgreSQL and prepare it by filling missing values.
    """
    df = load_data_from_postgres(query)
    df = clean_data(df)
    return df

def clean_data(df):
    """
    Clean the data by filling missing values and handling outliers.
    """
    # Fill missing values with 0 for numerical columns
    df['Dur. (ms)'] = pd.to_numeric(df['Dur. (ms)'], errors='coerce').fillna(0)
    df['Total DL (Bytes)'] = pd.to_numeric(df['Total DL (Bytes)'], errors='coerce').fillna(0)
    df['Total UL (Bytes)'] = pd.to_numeric(df['Total UL (Bytes)'], errors='coerce').fillna(0)
    
    # Additional cleaning steps can be added here if necessary
    return df

def perform_eda(df):
    """
    Perform exploratory data analysis to understand the dataset.
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values:\n", missing_values)
    
    # Summary statistics
    summary_stats = df.describe(include='all')
    print("Summary statistics:\n", summary_stats)
    
    # Visualize the distribution of handset types
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Handset Type', data=df, order=df['Handset Type'].value_counts().index)
    plt.title('Distribution of Handset Types')
    plt.xlabel('Count')
    plt.ylabel('Handset Type')
    plt.show()
    
    # Visualize the distribution of handset manufacturers
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Handset Manufacturer', data=df, order=df['Handset Manufacturer'].value_counts().index)
    plt.title('Distribution of Handset Manufacturers')
    plt.xlabel('Count')
    plt.ylabel('Handset Manufacturer')
    plt.show()

def identify_top_10_handsets(df):
    """
    Identify the top 10 handsets used by the customers.
    """
    top_10_handsets = df['Handset Type'].value_counts().head(10)
    return top_10_handsets

def identify_top_3_manufacturers(df):
    """
    Identify the top 3 handset manufacturers.
    """
    top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    return top_3_manufacturers

def identify_top_5_handsets_per_manufacturer(df, top_3_manufacturers):
    """
    Identify the top 5 handsets per top 3 handset manufacturer.
    """
    top_5_handsets_per_manufacturer = {}
    for manufacturer in top_3_manufacturers.index:
        top_5_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        top_5_handsets_per_manufacturer[manufacturer] = top_5_handsets
    return top_5_handsets_per_manufacturer

def plot_top_10_handsets(top_10_handsets):
    """
    Plot the top 10 handsets used by customers.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10_handsets.values, y=top_10_handsets.index, palette='viridis')
    plt.title('Top 10 Handsets Used by Customers')
    plt.xlabel('Number of Users')
    plt.ylabel('Handset Type')
    plt.show()

def plot_top_5_handsets_per_manufacturer(top_5_handsets_per_manufacturer):
    """
    Plot the top 5 handsets per top 3 manufacturers.
    """
    for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
        plt.figure(figsize=(10, 6))
        sns.barplot(x=handsets.values, y=handsets.index, palette='viridis')
        plt.title(f'Top 5 Handsets for {manufacturer}')
        plt.xlabel('Number of Users')
        plt.ylabel('Handset Type')
        plt.show()

def provide_recommendation(top_10_handsets, top_3_manufacturers, top_5_handsets_per_manufacturer):
    """
    Provide a detailed interpretation and recommendation to marketing teams.
    """
    recommendation = f"""
    Based on the analysis, the following insights have been identified:

    1. **Top 10 Handsets**:
    The top 10 handsets used by customers are:
    {top_10_handsets.to_string()}

    2. **Top 3 Handset Manufacturers**:
    The top 3 handset manufacturers are:
    {top_3_manufacturers.to_string()}

    3. **Top 5 Handsets per Top 3 Manufacturers**:
    The top 5 handsets for each of the top 3 manufacturers are:
    """
    for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
        recommendation += f"\n    - {manufacturer}:\n{handsets.to_string()}\n"

    recommendation += """
    **Recommendations for Marketing Teams**:
    - **Focus on Popular Handsets and Manufacturers**: Concentrate marketing efforts on the top 10 handsets and the top 3 manufacturers, as these are the most popular among customers. This can help in maximizing the impact of marketing campaigns.
    - **Targeted Promotions and Campaigns**: Develop targeted promotions and campaigns for these handsets and manufacturers to increase customer engagement and satisfaction. For example, special discounts, bundle offers, or loyalty programs can be designed around these popular handsets.
    - **Negotiation with Manufacturers**: Consider negotiating better deals with the top 3 manufacturers to offer exclusive promotions or discounts on their popular handsets. This can help in attracting more customers and increasing sales.
    - **Tailored Marketing Messages**: Use the insights from the top 5 handsets per manufacturer to tailor marketing messages and offers to specific customer segments who prefer these handsets. Personalized marketing can lead to higher conversion rates and customer loyalty.
    - **Monitor Trends**: Continuously monitor the trends in handset usage and manufacturer popularity to stay updated with customer preferences. This will help in making informed decisions for future marketing strategies.
    """
    print(recommendation)
