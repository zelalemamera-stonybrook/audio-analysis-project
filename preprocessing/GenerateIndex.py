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
	data contains a column for ipa, which has a sequence of phoemes that is broken up by a syllable marker.  in order to
	return the location of this marker,  we need to know the translated sequence as used by mfa model, these are the exact symbols used by the model to define intervals over the sound.
	'''
	data = pd.read_csv('./data/table.csv')
	data = data.set_index('Unnamed: 0')
	write_syllables_data(data)


def write_syllables_data(df: DataFrame):
	'''
	writes the index location of all the syllables for each word in this dataframe at ./data/syllable_index.txt
	'''
	print('writing syllable index')
	path = Path('./data/syllable_index.txt')
	with path.open(mode='w') as f:
		f.write('id	class	1	2	3	4\n')
		data = [(i, find_syllable_location(ipa)) for i, ipa in zip(df.index, df['ipa'])]
		for i, syll_list in data:
			if len(syll_list) == 2:
				syll1 = syll_list[-2]
				syll2 = syll_list[-1]
				f.write(f'{i}.TextGrid	2	{syll1}	{syll2}	0	0\n')
			elif len(syll_list) == 3:
				syll1 = syll_list[-3]
				syll2 = syll_list[-2]
				syll3 = syll_list[-1]
				f.write(f'{i}.TextGrid	3	{syll1}	{syll2}	{syll3}	0\n')
			elif len(syll_list) == 4:
				syll1 = syll_list[-4]
				syll2 = syll_list[-3]
				syll3 = syll_list[-2]
				syll4 = syll_list[-1]
				f.write(f'{i}.TextGrid	4	{syll1}	{syll2}	{syll3}	{syll4}\n')

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
	counter_list = counter_list[1:]
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
	additionaly, mfa strings are delimited by space, however the ipa provided does not come with space delimitation,
	this needs to be added with the additional complication of that any suprasegmatal sequences attatched to an ipa character should be considered a part of the same token.
	after this cleaning is applied, the ipa string is sent to the raw ipa to model symbols function to be transformed into an roughly equivalent sequence of an mfa phone string,
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
