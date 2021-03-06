# -*- coding: utf-8 -*-
"""
@author: bhass

process_data.py is an ETL (extract-transform-load) pipeline that extracts 
disaster messages and categories from csv files, cleans the data, and loads 
the clean data into a database. To run from the commmand prompt in the data 
directory, use:

python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

"""
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	"""
	load_data pulls in the disaster response messages and categories.csv files
	and merges them into a single dataframe
	
	ARGS:
		messages_filepath: path to messages csv file (eg. ../data/messages.csv)
		categories_filepath: path to categories csv file (eg. ../data/categories.csv)
	
	RETURNS:
		df: dataframe containing messages and corresponding categories
	"""
	messages = pd.read_csv(messages_filepath) # load messages dataset
	categories = pd.read_csv(categories_filepath) # load categories dataset
	df = messages.merge(categories,on='id') # merge messages and categories datasets
	
	return df

def clean_data(df):
	"""
	clean_data cleans the dataframe, including the following steps:
		1) renames categories to include only the category name
		2) converts categories to numbers 0 or 1
		3) drops duplicate messages
	
	ARGS:
		df: dataframe containing messages and categories, loaded from load_data
	
	RETURNS:
		df: dataframe containing cleaned messages, categories as 0 or 1 and 
		removed duplicates
	"""
	# create a dataframe of the 36 individual category columns
	categories = df.categories.str.split(';',expand=True)
	
	# select the first row of the categories dataframe
	row = categories.iloc[0]
	# use this row to extract a list of new column names for categories
	category_colnames = list(row.apply(lambda x: x[:-2]))
	# rename the columns of `categories`
	categories.columns = category_colnames
	
	# convert category values to just numbers 0 or 1
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].str[-1]
		# convert column from string to numeric
		categories[column] = pd.to_numeric(categories[column])
	
	#set upper limit to 1 - this will change all values of 2 in the related column to 1
	categories = categories.clip_upper(1)
	
	# replace categories column in df with new category columns
	# drop the original categories column from `df`
	df = df.drop(columns=['categories']) 
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df,categories],axis=1)
	#remove duplicates
	df = df.drop_duplicates(subset=['message'])
	
	return df

def save_data(df, database_filename):
	"""
	save_data saves the dataframe to a database (eg. DisasterResponse.db)
	
	ARGS:
		df: cleaned dataframe containing messages and categories, created from
		load_data and clean_data
	
	RETURNS:
		df: sql database with table "MessagesCategories"; name is specified by 
		input arg (eg. DisasterResponses.db)
	"""
	engine = create_engine('sqlite:///' + database_filename)
	df.to_sql('MessagesCategories', engine, index=False) 


def main():
	"""
	main loads reads in input arguments, loads messages and category csv files, 
	cleans the dataframe, and saves output to the database file.
	"""
	if len(sys.argv) == 4:

		messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

		print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
		.format(messages_filepath, categories_filepath))
		df = load_data(messages_filepath, categories_filepath)

		print('Cleaning data...')
		df = clean_data(df)

		print('Saving data...\n    DATABASE: {}'.format(database_filepath))
		save_data(df, database_filepath)

		print('Cleaned data saved to database!')

	else:
		print('Please provide the filepaths of the messages and categories '\
		'datasets as the first and second argument respectively, as '\
		'well as the filepath of the database to save the cleaned data '\
		'to as the third argument. \n\nExample: python process_data.py '\
		'disaster_messages.csv disaster_categories.csv '\
		'DisasterResponse.db')


if __name__ == '__main__':
	main()