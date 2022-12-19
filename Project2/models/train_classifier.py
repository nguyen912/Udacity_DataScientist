import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sqlite3
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

nltk.download(['punkt', 'wordnet'])


def load_data(db_path):
    """
    Loads data from SQLite database.

    Parameters:
    db_path: database file path

    Returns:
    X: Features
    Y: Target
    """
    # load data from database 
    engine = create_engine(f'sqlite:///{db_path}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y


def tokenize(text):
    """
    Tokenize text.

    Parameters:
    text: Text to be tokenized

    Returns:
    clean_words: Returns cleaned words
    """
    # tokenize text
    tokens = tokenize.word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_text = []
    for tok in tokens:
        # lemmatize, normalise case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_words.append(clean_tok)

    return clean_words


def build_model():
    """
    Builds classifier and tunes model using GridSearchCV.

    Returns:
    cv: Classifier 
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return cv


def evaluate_model(model, x_test, y_test):
    """
    Evaluates the performance of model and returns classification report. 

    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test

    Returns:
    Classification report for each column
    """
    y_pred = model.predict(x_test)
    for index, column in enumerate(y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, path):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(path, 'wb'))


def main():
    """ Builds the model, trains the model, evaluates the model, saves the model."""
    if len(sys.argv) == 3:
        db_path, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(db_path))
        X, Y = load_data(db_path)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()