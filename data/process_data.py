"""
Preprocessing of Data
Project: Disaster Response Pipeline
**Udacity - Data Science Nanodegree**

Sample Script Syntax:

python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>

Sample Script Execution:
python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db

Arguments Description:
    1) Path to the messages CSV file(e.g. disaster_messages.csv)
    2) Path to the categories CSV file(e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)
"""

import sys
import numpy as np
import pandas as pd
import os
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Messages Data, the Categories Data, and merge the 2

    Arguments:
        messages_filepath: Path to the messages CSV file
        categories_filepath: Path to the categories CSV file
    Output:
        df: Merged data containing messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df


def clean_data(df):
    """
    Clean the above Dataframe with category names and binarlized values

    Arguments:
        df: Merged data with messages and categories
    Outputs:
        df: above datafarme with categories cleaned up, values binalized
    """

    # Split the categories
    categories = df['categories'].str.split(pat=';',expand=True)

    #Fix the categories columns name
    row = categories.iloc[[0]]
    category_colnames = list(row.apply(lambda category_name: category_name.str.split('-')[0][0]))
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)

    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function

    Arguments:
        df: Combined data containing messages and categories with categories cleaned up
        database_filename: Path to SQLite destination database
    """

    engine = create_engine('sqlite:///'+ database_filename)
    table_name = os.path.basename(database_filename).replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    """
    The Main to start the data processing functions.
    There are three steps:
        1) Load Messages Data and Categories Data
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
