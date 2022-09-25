import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import nltk

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re


app = Flask(__name__)

#need to put the tokenize function here, otherwise, main cannot find it
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

# so does the StartingVerbExtractor class
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


# load data
engine = create_engine('sqlite:///../data/disaster_response_db.db')
df = pd.read_sql_table('disaster_response_db_table', engine)

# load model
model = joblib.load("../models/classification_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:,4:].columns
    category_counts = (df.iloc[:,4:]).sum().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # GRAPH 1 - genre distribution
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
            # GRAPH 2 - category distribution    
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 40
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()