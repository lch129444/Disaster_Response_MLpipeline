"""
Classifier Model training Process for 
Disaster Response Pipeline

Sample Script Syntax:
python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

To run ML pipeline that trains classifier and saves:
   python train_classifier.py ../data/disaster_response_db.db classification_model.pkl

Arguments:
    Path to SQLite destination database (../data/disaster_response_db.db)
    Path to pickle file name where ML model needs to be saved (classification_model.pkl)
"""

# import libraries
import nltk
import numpy as np
import pandas as pd
import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    """
    This function loads Data from the Database
    
    Input:
        database_filepath: Path to SQLite destination database (disaster_response_db.db)
    Output:
        X: a dataframe with messages as feature
        Y: a dataframe with labels
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    #Remove child alone since it is all 0
    df = df.drop(['child_alone'],axis=1)
    
    #Since no info shows the related ==2 should be categorized to 1 or 0, delete
    df = df[df.related !=2]
    
    # Extract X and y variables for the modelling
    X = df['message']
    y = df.iloc[:,4:]
    
    #used in main
    category_names = y.columns.values
    #print(X)
    #print(y.columns)
    return X, y, category_names


def tokenize(text):
    """
    Tokenize the text
    - replace url
    - word_tokenize
    - WordNetLemmatizer
    - lower
    - strip
    
    Arguments:
        text: Text message which needs to be tokenized
    Output:
        clean_tokens: List of tokens extracted/cleaned
    """
    # Replace urls with a urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    #remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]","",text)
    
    # Extract word tokens
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitize words
    lemmatizer = nltk.WordNetLemmatizer()

    # & normalize case and strip blanks
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    This class extracts the starting verb of a sentence,
    as a feature for the ML classifier
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
    # it will return "list index out of range" warning sometimes
            if len(pos_tags) > 0:
                first_word, first_tag = pos_tags[0]
                #print (first_word, first_tag)
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
            else:
                return 0
        return 0

    # Given it is a tranformer we can return the self 
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def build_model():
    """
    Build Pipeline function
    
    Output:
        A ML classification Pipeline that process text messages into categories.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters_grid = {
    'features__text_pipeline__count_vectorizer__ngram_range': ((1, 1), (1, 2)),
              'classifier__estimator__n_estimators': [50, 100, 200]}

    pipeline = GridSearchCV(pipeline, param_grid=parameters_grid, scoring='f1_micro', n_jobs=-1)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies trained ML pipeline to test and prints out the model performance
    
    Arguments:
        model: trained ML Pipeline (could be the pipeline from previous build_pipeline)
        X_test: Test features
        Y_test: Test labels
        category_names: label names (could be multi-output:category_names = y.columns.values)
    """
    Y_pred_test = model.predict(X_test)
    
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save trained model function as Pickle file,for later use.
    
    Arguments:
        model: GridSearchCV(could be the trained Pipeline)
        model_filepath: output path
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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