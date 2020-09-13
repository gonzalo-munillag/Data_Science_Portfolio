# Disaster Response Pipeline Project

### Project Summary

This project aims to create a data pipeline from loading to model evaluation to automatize future executions of the pipeline on different future values of datasets.

### Running Instructions Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files

1. disaster_messages.csv: Contains a dataset of the messages send by people in disaster zones
2. disaster_categories.csv: Contains the categories of the disasters that may happen on Earth
3. DisasterResponse.db: It is a sql dataset containing merged and cleaned data from the previous two datasets
4. classifier.pkl: contains a saved model trained with the DisasterResponse.db. It predicts to which category a message belongs
5. process_data.py: It contains the ETL pipleine 
6. train_classifier.py: It contains the code to train a classifer on the datasets formerly described

In the folder app, one may find the files to execute a web application where one can visualize the results of the model.

