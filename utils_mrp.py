import numpy as np
import os
import pandas as pd
import csv
import pickle

def checkFolders(folders):
	"""
	Check whether folders are present and creates them if necessary
	"""
	for folder in folders:
		if not os.path.exists(folder):
			print(f'Making directory {folder}')
			os.makedirs(folder)

def checkFiles(files):
	"""
	Check whether files are present and creates them if necessary
	"""
	if not type(files) == list:
		raise ValueError('Please provide checkFiles with a list')
	for file in files:
		if not os.path.exists(file):
			print(f'Making file {file}')
			open(file, 'a').close()

def writeColumns(file, settings):
	"""
	Writes the keys of dict settings as column names to csv file file
	"""
	with open(file, 'a') as f:
		wr = csv.writer(f, dialect='excel')
		wr.writerow(settings.keys())

def save_model_settings(nholder, settings):
	"""
	Save the model settings to a csv file. 
	
	#_______________________________________________
	nholder 			class 	holder class for IMNN
	settings 			dict 	all settings that should be saved
	
	"""

	file = nholder.modelsettings_name
	# open the previous modelsettings.csv file
	try:
		csv_settings = pd.read_csv(file)
	except pd.errors.EmptyDataError: 
		writeColumns(file,settings)
		csv_settings = pd.read_csv(file)

	appenddata = []
	for k in settings.keys():
		appenddata.append(settings[k])
	appenddata = np.array(appenddata)
	
	# Append to end of file
	appendloc = len(csv_settings) 
	csv_settings.loc[appendloc] = appenddata
	# Write 
	csv_settings.to_csv(file, index = False)

def save_final_fisher_info(nholder, n):
	"""
	Save final determinant of Fisher info on train and test set after training
	the network
	"""

	file = nholder.modelsettings_name
	# open the modelsettings.csv file
	csv_settings = pd.read_csv(file)
	# the row of the current version
	current_row = len(csv_settings)-1
	assert csv_settings['Version'][current_row] == nholder.modelversion

	csv_settings['Final detF train'][current_row] = '{0:.2f}'.format(n.history["det(F)"][-1])
	csv_settings['Final detF test'][current_row] = '{0:.2f}'.format(n.history["det(test F)"][-1])

	csv_settings.to_csv(file, index = False)

def save_history(nholder, n):
	"""
	Save the history of a network after training to the historydir as pickle file
	"""
	file = nholder.historydir+'history'+str(nholder.modelversion)+'.pkl'
	print (f'Saving History to file: {file}')
	with open(file, "wb") as f:
		pickle.dump(n.history,f)

def load_history():
	print ("TODO")