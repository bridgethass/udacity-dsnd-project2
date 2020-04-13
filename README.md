# udacity-dsnd-project2
Udacity Data Science Nanodegree Project 2 - Disaster Response Pipeline

### Table of Contents

1. [Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [Data Descriptions](#data)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions <a name="instructions"></a>

# Disaster Response Pipeline Project

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

This project ...

## Data Descriptions <a name="data"></a>

  1) `disaster_categories.csv`: 
  2) `disaster_messages.csv`: 

## File Descriptions <a name="files"></a>

The repository contains the following files:

    1) data/process_data.py
    2) models/train_classifier.py
    3) models/viz_results.py

## Results<a name="results"></a>

## Licensing, Authors, Acknowledgments <a name="licensing"></a>

Data is provided by [...](http://) The code here is licensed under open source GNU General Public License v3.0, and is free to use as you wish, with no guarantees :)
