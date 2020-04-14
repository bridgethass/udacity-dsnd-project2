# -*- coding: utf-8 -*-
"""
@author: bhass

train_classifier.py creates a classification model for the disaster response 
message from message and category data generated from data/process_data.py,
which cleans and saves the raw csv data into the DisasterResponse.db file

To run ML pipeline that trains classifier and saves model to classifier.pkl 
from command prompt, for example:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

"""

import os, sys, re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

#nltk 
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#sklearn 
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#pickle
import pickle
#%%
#visualize results functions (import functions from file viz_results.py)
from viz_results import *
#%%
def load_data(database_filepath):
	"""
	load_data pulls in clean data (produced from the ../data/process_data.py) 
	from the database (db) file located in the ../data directory, reads it into 
	a dataframe, and returns the features (X, messages) and target variables 
	(Y, disaster categories)

	ARGS:
	database_filepath - relative path and database filename (../data/DisasterResponses.db)

	RETURNS:
	X - feature variable (messages)
	Y - target variables (category of disaster response,n x 36 array of 0s and 1s)
	category_names - names of each of the categories
	"""
#	cwd = os.getcwd()
#	os.chdir(os.path.dirname(database_filepath)+'/')
#	os.chdir('../data/')
#	database_name = os.path.basename(database_filepath)
#	engine = create_engine('sqlite:///'+database_name)

	#configure engine
	engine = create_engine('sqlite:///'+database_filepath)
	#read the MessagesCategories table into a dataframe
	df = pd.read_sql_table('MessagesCategories',engine)
	#drop the child_alone column because this only has zeros 
	#(will throw error in some pipelines if there is no data for a column)
	df = df.drop(labels=['child_alone'],axis=1)
	#separate the message into the feature variable
	X = df['message'].values
	#pull out the target variables (categories)
	Y = df.iloc[:,4:].values
	#get the names of the categories
	category_names = df.columns[4:]
#	os.chdir(cwd)
	return X, Y, category_names

#%%
def tokenize(text):
	"""
	tokenize processes text data using nltk packages, carrying out the following transformations:
		1) normalizes the case (set everything to lower case) and strip whitespace
		2) for each token - lemmatize (strip to root form), and remove stop words

	ARGS:
	text: string of words (input features) to be tokenized

	RETURNS:
	tokens - tokenized text to be used in model (list)
	"""
	# normalize case and remove punctuation
	text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

	# tokenize
	tokens = word_tokenize(text)

	# lemmatize and remove stop words
	stop_words = stopwords.words("english")
	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

	return tokens

#%%
def build_model():
	"""
	build_model carries out the following steps:
		1) sets up a pipeline consisting of the sklearn text feature extractions:
		CountVectorizer and TfidfTransformer
		2) creates a multi-output random forest classifier, and
		3) uses GridSearchCV to select the optimal pipeline parameters
	
	ARGS: 
		none - this function just sets up the model, the arguments are applied when this model is fit

	RETURNS:
		cv - multi-class classifier pipeline fine-tuned using the best parameters 
		determined from the GridSearchCV
	"""
	#set up pipeline with text transformation features and 
	#multioutput random forest classifier model
	pipeline = Pipeline([
		('vect', CountVectorizer(tokenizer=tokenize)),
		('tfidf', TfidfTransformer()),
		('clf',  MultiOutputClassifier(RandomForestClassifier()))
	])

	#set up parameters for GridSearchCV, for sake of processing time
	#only include a few for the random forest classifier
	rf_parameters = {
		'clf__estimator__criterion': ['gini', 'entropy'],
		'clf__estimator__max_depth': [3, None],
		'clf__estimator__n_estimators': [10, 50, 100]
	}

	cv = GridSearchCV(pipeline, param_grid=rf_parameters)

	return cv

#%%
def evaluate_model(model, X_test, Y_test, category_names):
	"""
	evaluate_model uses the model to predict the test set, and then displays
	the classification report consisting of the precision, recall, f1-score and 
	support for each category, and saves a figure of the results to 
	classification_report.png
	
	ARGS: 
		model - 
		X_text - 
		Y_test - 
		category_names

	RETURNS: 
		print out of classification report scores for each category
		saves figure "classification_report.png" to working directory
	"""
	#predict the model on the test data
	Y_pred = model.predict(X_test)
	#display the classifiation report (includes precision, recall , f1-score, and support)
	print(classification_report(Y_test,Y_pred,target_names=category_names))
	
	#use the plot_classificaiton_report function from the viz_results.py module 
	#to plot the classification scores, save figure to png file
	plot_classification_report(classification_report(Y_test,Y_pred,target_names=category_names))
	plt.savefig('classification_report.png', dpi=200, format='png', bbox_inches='tight')

#%%
def save_model(model, model_filepath):
	"""
	save_model saves the model as a pickle file to the model_filepath
	
	ARGS: 
		model - model (pipeline) to save
		model_filepath - pickel model fileath and name

	RETURNS: 
		saves pickle file to model_filepath
	"""
	#write model to pickle file:
	pickle.dump(model, open(model_filepath, 'wb'))

#%%
def main():
	"""
	main pulls in system arguments from the command line executes the following:
		1) loads data from database and creates features and target variables
		2) splits the features and target variables into training and test sets
		3) builds the pipeline (model) with gridsearch cv
		4) fits the model to the training data
		5) evaluates the model (prints the classification report for each category)
		6) saves the model to a pickel file
	
	ARGS: 
		system arguments:
			train_classifier.py script & filepath: (models/train_classifier.py)
			database with clean disaster messages and categories: data/DisasterResponse.db 
			pickle_filepath: (eg. models/classifier.pkl)

	RETURNS: 
		prints classification report
		saves classification report figure to working directory (models)
		saves pickle file to model_filepath
	"""

	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]
		print('Loading data...\n    DATABASE: {}'.format(database_filepath))
		X, Y, category_names = load_data(database_filepath)
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

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