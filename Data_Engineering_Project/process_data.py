# import packages
import sys
import pandas as pd
import numpy as np


def load_data(data_file):

    # etl_pipeline.py
    # read in file
    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')

    # Merge data
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='left')
    
    # clean data

    # Extract column names
    # Extract the first value of the categories column as a string
    cat_names_series = pd.Series(df['categories'].iloc[0])[0]

    # Create list of categories by separating b ;
    sep = ';'
    cat_names_series = cat_names_series.split(';')

    # Eliminate the unnecessary part of the stirng after -
    callablet_names_series = pd.Series(cat_names_series).str.split('-', 1).apply(lambda x: x[0])

    # create a dataframe of the 36 individual category columns
    categories = pd.Series(df['categories']).str.split(pat=';',expand=True)
    categories.columns = cat_names_series

    # Convert values of the categories (0 or 1) into numeric
    for column in categories.columns:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str.get(-1)

    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True, axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # load to database
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql('InsertTableName', engine, index=False)

    # define features and label arrays
    X = df[messages.columns]
    y = df.drop(messages.columns, axis=1)

    return X, y


