import pandas as pd
from pandas.tseries.offsets import MonthEnd
import mysql.connector
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='.env')

# MySQL Database Configuration from .env
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
logger.info("DB INFO {}", DB_NAME)

def data_retrieval(table_name):
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USERNAME,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        if connection.is_connected():
            logger.info("Database connection established successfully.")
    except mysql.connector.Error as e:
        logger.error(f"Failed to connect to the database: {e}")
        connection = None

    try:
        # Fetch data directly from the database
        query = f"SELECT * FROM `{table_name}`"
        df = pd.read_sql(query, connection)
        df.to_csv('news_data.csv')

        return df
    except Exception as e:
        raise ValueError("Error fetching data': {e}")
    

