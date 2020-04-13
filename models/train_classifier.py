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
	cwd = os.getcwd()
	os.chdir(os.path.dirname(database_filepath))
	database_name = os.path.basename(database_filepath)
	engine = create_engine('sqlite:///'+database_name)
	df = pd.read_sql_table('MessagesCategories',engine)
	df = df.drop(labels=['child_alone'],axis=1)
	X = df['message'].values
	Y = df.iloc[:,4:].values
	category_names = df.columns[4:]
	os.chdir(cwd)
	return X, Y, category_names

#%%
def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

#%%
def build_model():
	pipeline = Pipeline([
		('vect', CountVectorizer(tokenizer=tokenize)),
		('tfidf', TfidfTransformer()),
		('clf',  MultiOutputClassifier(SGDClassifier(max_iter=1000),n_jobs=-1))
#		('clf',  MultiOutputClassifier(RandomForestClassifier()))
	])

#	parameters = {
#		'vect__ngram_range': ((1, 1), (1, 2)),
#		'vect__max_df': (0.5, 0.75, 1.0),
#		'vect__min_df': (0.05, 0.1),
#		'tfidf__norm': ('l1', 'l2'),
##		'clf__estimator__max_iter': (100,1000),
##		'clf__estimator__loss': ('hinge','log'),
#	}

	parameters = {
		'vect__ngram_range': ((1, 1), (1, 2)),
		'vect__max_df': (0.5, 0.75, 1.0),
		'vect__min_df': (0.05, 0.1),
		'tfidf__norm': ('l1', 'l2'),
		'clf__estimator__max_iter': (100,1000),
		'clf__estimator__loss': ('hinge','log'),
	}

	cv = GridSearchCV(pipeline, param_grid=parameters)

	return cv

#%%
def evaluate_model(model, X_test, Y_test, category_names):
	Y_pred = model.predict(X_test)
	print(classification_report(Y_test,Y_pred,target_names=category_names))
	plot_classification_report(classification_report(Y_test,Y_pred,target_names=category_names))

#%%
def save_model(model, model_filepath):
    pass

#%%
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