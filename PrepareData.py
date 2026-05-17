'''
The following program reads in spectral features and audio files from the subfolders in this directory and 
prepares the data in a format that is valid to be provided to StressClassifier neural network. 
'''
from pathlib import Path
import torchaudio
import torchcodec
import torch
import pandas as pd
import json

def binarize(i: int, n: int):
	'''
	returns a list of length n, with all zeros except at position i
	'''
	output = torch.zeros((n,))
	output[ i - 1] = 1
	return output.tolist()
	
	
	
if __name__ == '__main__':
	
	train_input_list = []
	train_gold_list = []
	train_path_list = ['wavefiles_syllabified/syllable_2/train'
				'wavefiles_syllabified/syllable_3/train'
				'wavefiles_syllabified/syllable_4/train'
				]
	for path in train_path_list:
		directory = Path(path)
		csv_path = next(directory.glob('*.csv'))
		dataframe = pd.read_csv(csv_path)
		for i in range(len(dataframe)):
			syllable_list = sorted(list(directory.glob(f'file{i}*.wav')))
			audio_list = []
			for syllable in syllable_list:
				audio_list.append((torchaudio.load(syllable)[0][0]).tolist())
			gold_list = binarize(dataframe['stress'][i], len(audio_list))
			train_input_list.append(audio_list)
			train_gold_list.append(gold_list)
	Path('train_data').mkdir()
	input = Path('train_data/input.json')
	input.touch()
	json.dump(train_input_list, input.open(mode='w'))
	gold = Path('train_data/gold.json')
	gold.touch()
	json.dump(train_gold_list, gold.open(mode='w'))
	
	test_input_list = []
	test_gold_list = []
	test_path_list = ['wavefiles_syllabified/syllable_2/test'
				'wavefiles_syllabified/syllable_3/test'
				'wavefiles_syllabified/syllable_4/test'
				]
	for path in test_path_list:
		directory = Path(path)
		csv_path = next(directory.glob('*.csv'))
		dataframe = pd.read_csv(csv_path)
		for i in range(len(dataframe)):
			syllable_list = sorted(list(directory.glob(f'file{i}*.wav')))
			audio_list = []
			for syllable in syllable_list:
				audio_list.append((torchaudio.load(syllable)[0][0]).tolist())
			gold_list = binarize(dataframe['stress'][i], len(audio_list))
			test_input_list.append(audio_list)
			test_gold_list.append(gold_list)
	Path('test_data').mkdir()
	input = Path('test_data/input.json')
	input.touch()
	json.dump(test_input_list, input.open(mode='w'))
	gold = Path('test_data/gold.json')
	gold.touch()
	json.dump(test_gold_list, gold.open(mode='w'))
	
	dev_input_list = []
	dev_gold_list = []
	dev_path_list = ['wavefiles_syllabified/syllable_2/dev'
				'wavefiles_syllabified/syllable_3/dev'
				'wavefiles_syllabified/syllable_4/dev'
				]
	for path in test_path_list:
		directory = Path(path)
		csv_path = next(directory.glob('*.csv'))
		dataframe = pd.read_csv(csv_path)
		for i in range(len(dataframe)):
			syllable_list = sorted(list(directory.glob(f'file{i}*.wav')))
			audio_list = []
			for syllable in syllable_list:
				audio_list.append((torchaudio.load(syllable)[0][0]).tolist())
			gold_list = binarize(dataframe['stress'][i], len(audio_list))
			dev_input_list.append(audio_list)
			dev_gold_list.append(gold_list)
	Path('dev_data').mkdir()
	input = Path('dev_data/input.json')
	input.touch()
	json.dump(dev_input_list, input.open(mode='w'))
	gold = Path('dev_data/gold.json')
	gold.touch()
	json.dump(dev_gold_list, gold.open(mode='w'))
	
			
				
			
			
		
	
		
		
	
	