import sys
# Pandas, numpy, SQL engine
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

def load_data(database_filepath): 
        
    """ Loads a swl dataset and outputs the features and the labels.
    
    INPUT
    database_filepath (str) -- filepath to sql dataset
    
    OUTPUT
    X (array) -- features
    y (array) -- labels
    category_names (array) -- names of the classification buckets 
    """
    
    # Dumping the sql dataset into a dataframe
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath.split('/')[-1], engine)

    # Selecting the data for our features and labels
    X = df['message'].values
    y = df.drop(['id', 'message', 'original', 'genre', 'related'], axis=1)
    category_names = y.columns
    
    return X, y, category_names 

def tokenize(text):
    
    """ Tokenizes a text into words, considering: replace urls, punctuation, stopwords, and lemmatization.
        It will be used in a transformer inside the pipelie
    
    INPUT
    text (array) -- array containing text on each of its elements
    
    OUTPUT
    clean_tokens (array) -- array containing clean text on each of its elements
    """
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
       
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """ It builds the model to be used for predictions
    
    OUTPUT
    cv (model object) -- model to be used for training
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__max_depth': [None, 5, 10],
    'clf__estimator__min_samples_leaf': [1,3]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ It evaluates how performant is the model
    
    INPUT
    model (model object) -- model to be evaluated
    X_test (dataframe of features) - It contains the features the model will use for prediction
    y_test (dataframe of labels) -- It contains the true value of the predictions
    category_bnames (array) -- It contains an array of labels of the classes on which data will be classified
    """
    
    # We calculate the accuracy
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()
    print('The accuracy of the model is {} \n'.format(accuracy))
          
    # We calculate the recall, precision and f1 score
    report = classification_report(Y_test.values, y_pred, target_names=category_names)
    print('The precision, recall and f1-score are: \n')
    print(report)

def save_model(model, model_filepath):
    
    """ It evaluates how performant is the model
    
    INPUT
    model (model object) -- model to be evaluated
    model_filepath (str) -- Path where the model will be saved
    """
    
    # Save model to disk
    pickle.dump(model, open(model_filepath, 'wb'))

def load_model(model_filepath):
    
    """ Loads the model from disk
    
    INPUT
    model (model object) -- model to be evaluated
    
    OUTPUT
    model_filepath (str) -- Path where the model will be loaded from
    """
        
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
        
    return model


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
