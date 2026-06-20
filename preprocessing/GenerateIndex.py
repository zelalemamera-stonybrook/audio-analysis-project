'''
The following python program generates one of the components required for an automatic syllabification, which is the syllable locations of each file, this is extracted from the syllabified phoneme sequence
'''

import pandas as pd
from pandas import DataFrame
from pathlib import Path
import json
import re

path = Path('./data/ipa_to_mfa.json')
model_symbols = {}
with path.open(mode='r') as f:
		model_symbols = json.load(f)
		
def write_syllable_table():
	'''
	each table of the data contains a column for ipa, which has a sequence of phoemes that is broken up by a syllable boundary.  in order to 
	return the location of this boundary, we need to know the translated sequence as used by mfa model, as these are the exact symbols used by the model to define intervals over the sound.
	'''
	data_2 = pd.read_csv('./data/data_2.csv')
	data_3 = pd.read_csv('./data/data_3.csv')
	data_4 = pd.read_csv('./data/data_4.csv')
	write_syllables_data_2(data_2)
	write_syllables_data_3(data_3)
	write_syllables_data_4(data_4)
	

def write_syllables_data_2(df: DataFrame):
	'''
	writes the index location of all the syllables for each word in this two syllable dataframe at ./data/data_2/syllable_index.txt
	'''
	print('writing two syllable index')
	path = Path('./data/data_2/syllable_index.txt')
	with path.open(mode='w') as f:
		f.write('filename\tsyll1\n')
		data = [(i, find_syllable_location(ipa)) for i, ipa in zip(df.index, df['ipa'])]
		for i, syll_list in data:
			syll1 = syll_list[-1]
			f.write(f'file_{i}.TextGrid\t{syll1}\n')
			
def write_syllables_data_3(df: DataFrame):
	'''
	writes the index location of all the syllables for each word in this three syllable dataframe at ./data/data_3/syllable_index.txt
	'''
	path = Path('./data/data_3/syllable_index.txt')
	with path.open(mode='w') as f:
		f.write('filename\tsyll1\tsyll2\n')
		data = [(i, find_syllable_location(ipa)) for i, ipa in zip(df.index, df['ipa'])]
		for i, syll_list in data:
			syll1 = syll_list[-2]
			syll2 = syll_list[-1]
			f.write(f'file_{i}.TextGrid\t{syll1}\t{syll2}\n')
			
def write_syllables_data_4(df: DataFrame):
	'''
	writes the index location of all the syllables for each word in this four syllable dataframe at ./data/data_4/syllable_index.txt
	'''
	path = Path('./data/data_4/syllable_index.txt')
	with path.open(mode='w') as f:
		f.write('filename\tsyll1\tsyll2\tsyll3\n')
		data = [(i, find_syllable_location(ipa)) for i, ipa in zip(df.index, df['ipa'])]
		for i, syll_list in data:
			syll1 = syll_list[-3]
			syll2 = syll_list[-2]
			syll3 = syll_list[-1]
			f.write(f'file_{i}.TextGrid\t{syll1}\t{syll2}\t{syll3}\n')

def find_syllable_location(ipa: str):
	'''
	returns an integer representing the number of model symbols that preceede the occurence of each syllable marker
	'''
	print('received raw ipa', ipa)
	syllables = syllabify(ipa)
	counter_list = [0]
	for syll in syllables:
		model = map_to_mfa(syll)
		counter_list.append(len(re.split(r'\s', model)) + counter_list[-1])
	counter_list = counter_list[1:-1]
	print('generated syllable indices', counter_list)
	return counter_list
	
	
def syllabify(ipa: str):
	'''
	returns all of syllables of this ipa as a list assuming it is broken up by the syllable marker and the stress marker.
	'''
	stress_marker = 'ˈ'
	ipa = re.sub(r'\s', '', ipa)
	substrings = re.split(r'\.', ipa)
	syllable_list = []
	for substring in substrings:
		if substring[0] == stress_marker and substring[-1] != stress_marker:
			syllable_list += re.split(f"{stress_marker}", substring)[1:]
		elif substring[-1] == stress_marker and substring[0] != stress_marker:
			syllable_list += re.split(f"{stress_marker}", substring)[0:-1]
		elif substring[0] == substring[-1] and substring[0] == stress_marker:
			syllable_list += re.split(f"{stress_marker}", substring)[1:-1]
		else:
			syllable_list += re.split(f"{stress_marker}", substring)
	print('syllabified ipa into', syllable_list)
	return syllable_list

def map_to_mfa(ipa: str):
	'''
	the raw ipa strings are not sutiatble for alignment before putting them into a specific format; any suprasegmentals need to be removed if they are not a part of mfa vocabulary,
	additionaly, mfa strings are delimited by space, however the ipa provided does not come with space delimitation, this needs to be added with the additional comlplication of that any suprasegmatal sequences attatched to an 
	ipa character should be considered a part of the same token. after this cleaning is applied, the ipa string is sent to the raw ipa to model symbols function to be trnasformed into an roughly equivalent sequence of an mfa phone string, 
	which can be used for alignment directly.
	'''
	print('mapping syllable', ipa, 'to model symbols')
	ipa = re.sub(r"\s", "", ipa)
	unwanted = set([' ', ')', '(', '‿', 'ˌ', 'ˈ', '.', '͡'])
	suprasegmentals = set(['ˤ', 'ː'])
	ipa_list = list(ipa)
	cleaned_list = []
	segmented_list = []
	for ipa in ipa_list:
		if ipa not in unwanted:
			cleaned_list.append(ipa)
	for i, ipa in enumerate(cleaned_list):
		if ipa in suprasegmentals:
			segmented_list[-1] = f"{cleaned_list[i - 1]}{ipa}"
		else:
			segmented_list.append(ipa)
	for i, ipa in enumerate(segmented_list):
		print('mapping', segmented_list[i], 'to', model_symbols[ipa])
		segmented_list[i] = model_symbols[ipa]
	mapped = " ".join(segmented_list)
	print('mapping obtained', mapped)
	return mapped
    
    
    
if __name__ == '__main__':
	write_syllable_table()
	
	
	
	
	
	
	
	
	
	
	
	
	
	