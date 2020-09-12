# import packages
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# NLP
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ML 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import pickle

#Constants
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(data_file_path, database_name):

    """
    Loads data from datasets and wrangles them for later model training.

    INPUT
    data_file_path (str) -- path to data for extracting features and labels from sql.dataset
    database_name (str) -- Name of the dataset to load
    
    OUTPUT
    X(array) -- features 
    y(array) -- labels 
    """
        
    # read in file
    # 'sqlite:///DisasterResponse.db'
    # 'DisasterResponse.db'
    engine = create_engine(data_file_path)
    df = pd.read_sql_table(database_name, engine)

    # clean data


    # load to database


    # define features and label arrays


    return X, y


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return model_pipeline


def train(X, y, model):
    # train test split


    # fit model


    # output model test results


    return model


def export_model(model):
    # Export model as a pickle file



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
