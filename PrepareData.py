'''
The following program reads in spectral features and audio files from the subfolders in this directory and 
prepares the data in a format that is valid to be provided to StressClassifier neural network. 
'''
from pathlib import Path
import torchaudio
import torchcodec
import torch
from torch import Tensor
import pandas as pd
from pandas import DataFrame
import json
import PrepareSpectralFeatures

def binarize(i: int, n: int):
	'''
	returns a list of length n, with all zeros except at position i
	'''
	output = torch.zeros((n,))
	output[ i - 1] = 1
	return output.tolist()

def pad(vec: Tensor, n):
	'''
	pads the input Tensor to a size of n. Our data contains syllables of a various sizes, with the maximum length being 27,000, and
	the minimum being 4000. With the intention of not cutting out any data, and evening out the input dimention, we choose to pad all vectors 
	to a length of 30,000. Zero padding is applied to left and right ends. other sizes may be specified as well. Vec is assumed to be a one dimensional tensor,
	i.e. there should not be any channels. 
	input shape: (k,)
	output shape: (n, )
	'''
	diff = n - vec.shape[0]
	pad = torch.zeros((diff // 2,))
	first = pad.tolist()
	first += vec.tolist()
	first += pad.tolist()
	if diff % 2 == 1:
		first += [0]
	return torch.Tensor(first)
	
def mean(vector_list: list):
	'''
	returns the average value of this list, we assume that the sample is drawn from a uniform distribution.
	'''
	sum = 0
	for vec in vector_list:
		sum += vec
	return sum * (1 / len(vector_list))

def variance(vector_list: list):
	'''
	returns the expected value of (x - mean(x))^2
	'''
	meanx = mean(vector_list)
	x_squared = []
	for vec in vector_list:
		x_squared.append(vec**2)
	x_squared = torch.Tensor(x_squared)
	meanx2 = mean(x_squared)
	t = torch.zeros(meanx.shape)
	return meanx2 - (torch.addcmul( t, meanx, meanx))
	
	
	
def normalize_and_pad(output_list: list):
	'''
	the list in question is a sequence of words. each word is a list of syllables, for whom we have extracted a feature. this feature needs to be normalized over the whole set of syllables.
	then it is zero padded to 1k
	input shape: [w1,...,wn] where wi = [s1, s2, s3], [s1, s2], or [s1,s2,s3,s4]
	output shape: input shape
	the order of the sequence of words and syllables must be preserved.
	'''
	vector_list = []
	word_size = []
	for word in output_list:
		word_size.append(len(word))
	for word in output_list:
		for vec in word:
			vector_list.append(torch.Tensor(vec))
	tensor = torch.Tensor(vector_list)
	mean = mean(tensor)
	std = variance(tensor) ** 0.5
	t = torch.zeros(tensor.shape)
	normalized = torch.addcdiv(t, (tensor - mean) , std)
	embedded= []
	for syll in normalized:
		embedded.append(pad(syll, 990).tolist())
	word_list = []
	size = 0
	for n in word_size:
		word_list.append(embedded[size: size + n])
		size += n
	return word_list
		
	
			
def write_data(data_dict: dict):
	'''
	data_dict contains all of the raw spectral features. This function collects the relevant batch for every feature, and normalizes over the whole set, embeds each into a size of 1000, then saves each of the batches as a list of vectors
	input shape: dict
	output shape: [ f1, f2, ..., fh] where fi = [ w1, ..., wm]
	'''
	syllable_list = ['syllable_2', 'syllable_3', 'syllable_4']
	batch_list = ['train', 'test', 'dev']
	number_of_features = len(data_dict['syllable_2']['train'])
	features_train = []
	features_test  = []
	features_dev = []
	for i in range(number_of_features):
		output_list = []
		batch_sizes = []
		for batch in batch_list:
			size = 0
			for syllable in syllable_list:
				word_list = data_dict[syllable][batch][i]
				output_list += word_list
				size += len(word_list)
			batch_sizes.append(size)
		embedded_output = normalize_and_pad(output_list)
		train_size, test_size, dev_size = batch_sizes[0], batch_sizes[1], batch_sizes[-1]
		train_list, test_list, dev_list = embedded_output[:train_size], embedded_output[train_size: train_size + test_size], embedded_output[train_size + test_size:]
		features_train.append(train_list)
		features_test.append(test_list)
		features_dev.append(dev_list)
	path1 = Path('train_features.json')
	path2 = Path('test_features.json')
	path3 = Path('dev_features.json')
	path1.touch()
	path2.touch()
	path3.touch()
	json.dump(features_train, path1.open(mode='w'))
	json.dump(features_test, path2.open(mode='w'))
	json.dump(features_dev, path3.open(mode='w'))			
			
	
def prepare_data(batch: str):
	'''
	for the particular batch specified, write the input and gold pairs to the current directory
	'''
	input_list = []
	gold_list = []
	path_list = [f'wavefiles_syllabified/syllable_2/{batch}',
				f'wavefiles_syllabified/syllable_3/{batch}',
				f'wavefiles_syllabified/syllable_4/{batch}'
				]
	for path in path_list:
		directory = Path(path)
		csv_path = next(directory.glob('*.csv'))
		dataframe = pd.read_csv(csv_path)
		for i in range(len(dataframe)):
			word = sorted(list(directory.glob(f'file{i}*.wav')))
			padded_word = []
			for syllable in word:
				audio, samplerate = torchaudio.load(syllable)
				padded_audio = pad(audio[0], 30000)
				padded_word.append(padded_audio.tolist())
			input_list.append(padded_word)
			gold_list.append(binarize(dataframe['stress'][i], len(word)))
	Path(f'{batch}_data').mkdir()
	input = Path(f'{batch}_data/input.json')
	input.touch()
	json.dump(input_list, input.open(mode='w'))
	gold = Path(f'{batch}_data/gold.json')
	gold.touch()
	json.dump(gold_list, gold.open(mode='w'))
	
	
if __name__ == '__main__':
	prepare_data('train')
	prepare_data('test')
	prepare_data('dev')
	
	PrepareSpectralFeatures.prepare_data()
	
	path = Path('spectral_dictionary.json')
	data_dict = json.load(path.open(mode='r'))
	write_data(data_dict)
			
				
			
			
		
	
		
		
	
	