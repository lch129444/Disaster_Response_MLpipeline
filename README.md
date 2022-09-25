# Disaster Response Pipeline Project

## Instructions:
This Project is an assignment of Data Science Nanodegree Program by Udacity. The dataset provided by Udacity includes pre-labelled messages from social media apps during real-life disaster events. The goal is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.  The loading page is a distribution of the training dataset.

This repository includes the following steps:
1. Process raw data with an ETL pipeline to extract data from source, clean the data, add category-label and save to a SQLite DB
2. Build a NLP machine learning pipeline to train the cleaned dataset about different categories.  Save the model to a pickle file
3. Run an app and can load the model pickle file.  So, when new messages were put in the message bar, use the model to categorize in real time.

### Requirements
* Python 3.5+
* Date cleaning: Pandas, NumPy, SQLalchemy
* Machine Learning Libraries: Sciki-Learn
* Natural Language Process Libraries: NLTK
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

### Executing:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response_db.db models/classification_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing the Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model
