'''
The following program loads in all of the generated spectral features for the soundfiles and generates vectors for each sound file. 
'''
import pandas as pd
from pandas import DataFrame
import re
import torch
from torch import Tensor
import json
from pathlib import Path
    
    
    
def extract_number(filename: str):
    '''
    helper function extracts a number from the filename
    '''
    p = re.compile(r'\d')
    return int("".join(p.findall(filename)))
    
    
def clean_tables(df_dict):
    '''
    drops all the nans from table, sorts them first by their file name index, then by their syllable location
    '''
    key_list = ['syllable_2', 'syllable_3', 'syllable_4']
    key_list2 = ['train', 'test', 'dev']
    for key in key_list:
        for key2 in key_list2:
            df_list = df_dict[key][key2]
            new_list = []
            for dataframe in df_list:
                dataframe.dropna(inplace=True)
                dataframe_key = [extract_number(string) for string in dataframe['fileName']]
                dataframe['key'] = dataframe_key
                dataframe.sort_values(by=['key', 'name'], inplace=True)
                new_index = [i for i in range(len(dataframe))]
                dataframe.set_index(pd.Index(new_index), inplace=True)
                new_list.append(dataframe.drop(columns=['key', 'filename', 'name']))
            df_dict[key][key2] = new_list
    return df_dict
   
				
def collect_words(df: DataFrame, syllable: str):
	'''
	turns the rows in the dataframe into a list of words. this represents the feature encoding of a word of the specified syllable size 
	input shape: n * k where k is the number of words and n is syllables
	output shape: k 
	'''	
	key = {'syllable_2': 2, 'syllable_3':3, 'syllable_4':4}
	n = key[syllable]
	lst = []
	for i in range( len(df) / n):
		lst += torch.full((n,), i).tolist()
	df['key'] = lst
	feature_list = []
	for i in range(len(df) / n):
		word_features = df[df['key'] == i]
		feature_list.append(word_features.to_numpy().tolist())
	return feature_list
    
def write_spectral_features(df_dict: dict):
	'''
	for each feature, it generates the list of words that are represented by that feature. 
	input shape: dict[str:dict[str:list]]
	output shape: [f1,..., fn] where each fi = [w1, ..., wm]
	'''
	new_dict = {'syllable_2':{}, 'syllable_3':{}, 'syllable_4':{}}
	clean_dict = clean_tables(df_dict)
	batches= ['train', 'test', 'dev']
	syllables = ['syllable_2', 'syllable_3', 'syllable_4']
	for batch in batches:
		for syllable in syllables:
			new_dict[syllable][batch] = []
			feature_list = clean_dict[syllable][batch]
			for feature_df in feature_list:
				new_dict[syllable][batch].append(collect_words(feature_df, syllable))
	path = Path('spectral_dictionary.json')
	path.touch()
	json.dump(new_dict, path.open(mode='w'))
	

def prepare_data():
	'''
	main function which prepares all of the spectral features and writes them as a json dictionary to the current directory.
	'''
	

	train_formant_2_syll_table = pd.read_csv('spectral_data/syllable_2/train/duration_formant.csv')
	train_duration_2_syll_table = pd.concat([train_formant_2_syll_table['duration'], train_formant_2_syll_table['filename'], train_formant_2_syll_table['name']], axis=1)
	train_formant_2_syll_table.drop(columns='duration', inplace=True)
	
	train_intensity_2_syll_table = pd.read_csv('spectral_data/syllable_2/train/duration_intensity.csv')
	train_intensity_2_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
	
	train_pitch_2_syll_table = pd.read_csv('spectral_data/syllable_2/train/duration_pitch.csv')
	train_pitch_2_syll_table.drop(columns='duration', inplace=True)

	dev_formant_2_syll_table = pd.read_csv('spectral_data/syllable_2/dev/duration_formant.csv')
	dev_duration_2_syll_table = pd.concat([dev_formant_2_syll_table['duration'], dev_formant_2_syll_table['filename'],dev_formant_2_syll_table['name']], axis=1)
	dev_formant_2_syll_table.drop(columns='duration', inplace=True)
	
	dev_intensity_2_syll_table = pd.read_csv('spectral_data/syllable_2/dev/duration_intensity.csv')
	dev_intensity_2_syll_table.drop(columns=['duration', 'intensity1', 'intensity10'], inplace=True)
	
	dev_pitch_2_syll_table = pd.read_csv('spectral_data/syllable_2/dev/duration_pitch.csv')
	dev_pitch_2_syll_table.drop(columns='duration', inplace=True)

	test_formant_2_syll_table = pd.read_csv('spectral_data/syllable_2/test/duration_formant.csv')
	test_duration_2_syll_table = pd.concat([test_formant_2_syll_table['duration'], test_formant_2_syll_table['filename'],test_formant_2_syll_table['name']], axis=1)
	test_formant_2_syll_table.drop(columns='duration', inplace=True)
	
	test_intensity_2_syll_table = pd.read_csv('spectral_data/syllable_2/test/duration_intensity.csv')
	test_intensity_2_syll_table.drop(columns=['duration', 'intensity1', 'intensity10'], inplace=True)
	
	test_pitch_2_syll_table = pd.read_csv('spectral_data/syllable_2/test/duration_pitch.csv')
	test_pitch_2_syll_table.drop(columns='duration', inplace=True)
	
	train_formant_3_syll_table = pd.read_csv('spectral_data/syllable_3/train/duration_formant.csv')
	train_duration_3_syll_table = pd.concat([train_formant_3_syll_table['duration'], train_formant_3_syll_table['filename'], train_formant_3_syll_table['name']], axis=1)
	train_formant_3_syll_table.drop(columns='duration', inplace=True)
	
	train_intensity_3_syll_table = pd.read_csv('spectral_data/syllable_3/train/duration_intensity.csv')
	train_intensity_3_syll_table.drop(columns=['duration', 'intensity1', 'intensity10'], inplace=True)
	
	train_pitch_3_syll_table = pd.read_csv('spectral_data/syllable_3/train/duration_pitch.csv')
	train_pitch_3_syll_table.drop(columns='duration', inplace=True)

	dev_formant_3_syll_table = pd.read_csv('spectral_data/syllable_3/dev/duration_formant.csv')
	dev_duration_3_syll_table = pd.concat([dev_formant_3_syll_table['duration'],dev_formant_3_syll_table['filename'],dev_formant_3_syll_table['name']], axis = 1)
	dev_formant_3_syll_table.drop(columns='duration', inplace=True)
	
	dev_intensity_3_syll_table = pd.read_csv('spectral_data/syllable_3/dev/duration_intensity.csv')
	dev_intensity_3_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
	
	dev_pitch_3_syll_table = pd.read_csv('spectral_data/syllable_3/dev/duration_pitch.csv')
	dev_pitch_3_syll_table.drop(columns='duration', inplace=True)
	
	test_formant_3_syll_table = pd.read_csv('spectral_data/syllable_3/test/duration_formant.csv')
	test_duration_3_syll_table = pd.concat([test_formant_3_syll_table['duration'],test_formant_3_syll_table['filename'],test_formant_3_syll_table['name']], axis=1)
	test_formant_3_syll_table.drop(columns='duration', inplace=True)
	
	test_intensity_3_syll_table = pd.read_csv('spectral_data/syllable_3/test/duration_intensity.csv')
	test_intensity_3_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
	
	test_pitch_3_syll_table = pd.read_csv('spectral_data/syllable_3/test/duration_pitch.csv')
	test_pitch_3_syll_table.drop(columns='duration', inplace=True)
	
	train_formant_4_syll_table = pd.read_csv('spectral_data/syllable_4/train/duration_formant.csv')
	train_duration_4_syll_table = pd.concat([train_formant_4_syll_table['duration'],train_formant_4_syll_table['filename'],train_formant_4_syll_table['name']], axis=1)
	train_formant_4_syll_table.drop(columns='duration', inplace=True)
	
	train_intensity_4_syll_table = pd.read_csv('spectral_data/syllable_4/train/duration_intensity.csv')
	train_intensity_4_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
	
	train_pitch_4_syll_table = pd.read_csv('spectral_data/syllable_4/train/duration_pitch.csv')
	train_pitch_4_syll_table.drop(columns='duration', inplace=True)

	dev_formant_4_syll_table = pd.read_csv('spectral_data/syllable_4/dev/duration_formant.csv')
	dev_duration_4_syll_table = pd.concat([dev_formant_4_syll_table['duration'],dev_formant_4_syll_table['filename'],dev_formant_4_syll_table['name']], axis=1)
	dev_formant_4_syll_table.drop(columns='duration', inplace=True)
	
	dev_intensity_4_syll_table = pd.read_csv('spectral_data/syllable_4/dev/duration_intensity.csv')
	dev_intensity_4_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
	
	dev_pitch_4_syll_table = pd.read_csv('spectral_data/syllable_4/dev/duration_pitch.csv')
	dev_pitch_4_syll_table.drop(columns='duration', inplace=True)
	
	test_formant_4_syll_table = pd.read_csv('spectral_data/syllable_4/test/duration_formant.csv')
	test_duration_4_syll_table = pd.concat([test_formant_4_syll_table['duration'],test_formant_4_syll_table['filename'],test_formant_4_syll_table['name']], axis=1)
	test_formant_4_syll_table.drop(columns='duration', inplace=True)
	
	test_intensity_4_syll_table = pd.read_csv('spectral_data/syllable_4/test/duration_intensity.csv')
	test_intensity_4_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
	
	test_pitch_4_syll_table = pd.read_csv('spectral_data/syllable_4/test/duration_pitch.csv')
	test_pitch_4_syll_table.drop(columns='duration', inplace=True)
	
	df_dict = {
          	 'syllable_2': {
	 'train':[train_duration_2_syll_table, train_formant_2_syll_table, train_intensity_2_syll_table, train_pitch_2_syll_table],
	 'dev':[dev_duration_2_syll_table, dev_formant_2_syll_table, dev_intensity_2_syll_table, dev_pitch_2_syll_table],
	 'test':[test_duration_2_syll_table, test_formant_2_syll_table, test_intensity_2_syll_table, test_pitch_2_syll_table]
	 },
            	'syllable_3':{
	'train':[train_duration_3_syll_table, train_formant_3_syll_table, train_intensity_3_syll_table, train_pitch_3_syll_table],
	'dev':[dev_duration_3_syll_table, dev_formant_3_syll_table, dev_intensity_3_syll_table, dev_pitch_3_syll_table],
	'test':[test_duration_3_syll_table, test_formant_3_syll_table, test_intensity_3_syll_table, test_pitch_3_syll_table]
	},
            	'syllable_4':{
	'train':[train_duration_4_syll_table, train_formant_4_syll_table, train_intensity_4_syll_table, train_pitch_4_syll_table],
	'dev':[dev_duration_4_syll_table, dev_formant_4_syll_table, dev_intensity_4_syll_table, dev_pitch_4_syll_table],
	'test':[test_duration_4_syll_table, test_formant_4_syll_table, test_intensity_4_syll_table, test_pitch_4_syll_table]
	}
	}
		
	write_spectral_features(df_dict)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	