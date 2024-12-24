import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from a .env file to retrieve database connection details
load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_data_from_postgres(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Establish a connection to the PostgreSQL database using psycopg2
        connection = psycopg2.connect(
            host=DB_HOST,            # Host of the database server
            port=DB_PORT,            # Port where the database is listening (default is 5432)
            database=DB_NAME,        # Database name
            user=DB_USER,            # Username for database authentication
            password=DB_PASSWORD     # Password for database authentication
        )

        # Use pandas to execute the SQL query and return the result as a DataFrame
        df = pd.read_sql_query(query, connection)

        # Close the database connection after the query execution
        connection.close()

        # Return the DataFrame containing the query results
        return df

    except Exception as e:
        # In case of any error, print the exception message and return None
        print(f"An error occurred: {e}")
        return None


def load_data_using_sqlalchemy(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query using SQLAlchemy.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Create a connection string using the environment variables
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

        # Create an SQLAlchemy engine using the connection string
        engine = create_engine(connection_string)

        # Use pandas to execute the SQL query and return the result as a DataFrame
        df = pd.read_sql_query(query, engine)

        # Return the DataFrame containing the query results
        return df

    except Exception as e:
        # In case of any error, print the exception message and return None
        print(f"An error occurred: {e}")
        return None
