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
import os
import shutil
import subprocess

def binarize(i: int, n: int):
	'''
	returns a list of length n, with all zeros except at position i
	'''
	output = torch.zeros((n,))
	output[ i - 1] = 1
	return output.tolist()

def pad(vec: Tensor, n:int):
	'''
	pads the input Tensor to a size of n. Our data contains syllables of a various sizes, with the maximum length being 27,000, and
	the minimum being 4000. With the intention of not cutting out any data, and evening out the input dimention, we choose to pad all vectors 
	to a length of 30,000. Zero padding is applied to left and right ends. other sizes may be specified as well. Vec is assumed to be a one dimensional tensor,
	i.e. there should not be any channels. 
	input shape: (k,)
	output shape: (n, )
	'''
	print('padding vector...')
	print('input shape', vec.shape)
	if vec.shape == torch.tensor([1])[0].shape:
		vec = torch.tensor([vec])
	diff = n - len(vec)
	if diff <  0 :
		print('cutting vector by', diff)
		cut = - diff // 2
		temp = vec[cut:-cut]
		if diff % 2 == 1:
			return temp[1:]
		return temp
	print('zeros to be added', diff)
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
		x_squared.append((vec**2).tolist())
	x_squared = torch.Tensor(x_squared)
	meanx2 = mean(x_squared)
	t = torch.zeros(meanx.shape)
	return meanx2 - (torch.addcmul( t, meanx, meanx))
	
	
def mean_pad(vec: Tensor, n:int):
	'''
	pads the input Tensor to a size of n, with the slight adjustment that the values used to pad the vector are its mean. 
	'''
	print('padding vector...')
	print('input shape', vec.shape)
	if vec.shape == torch.tensor([1])[0].shape:
		vec = torch.tensor([vec])
	diff = n - len(vec)
	if diff <  0 :
		print('cutting vector by', diff)
		cut = - diff // 2
		temp = vec[cut:-cut]
		if diff % 2 == 1:
			return temp[1:]
		return temp
	print('means to be added', diff)
	mean = torch.mean(vec).item()
	pad = torch.full((diff // 2,), mean)
	first = pad.tolist()
	first += vec.tolist()
	first += pad.tolist()
	if diff % 2 == 1:
		first += [mean]
	return torch.Tensor(first) * 10

	
def normalize_and_pad(output_list: list):
	'''
	the list in question is a sequence of words. each word is a list of syllables, for whom we have extracted a feature. this feature needs to be normalized over the whole set of syllables.
	then it is zero padded to 990
	input shape: [w1,...,wn] where wi = [s1, s2, s3], [s1, s2], or [s1,s2,s3,s4]
	output shape: input shape
	the order of the sequence of words and syllables must be preserved.
	'''
	vector_list = []
	word_size = []
	for word in output_list:
		word_size.append(len(word))
	print('word sizes found', word_size)
	for word in output_list:
		for vec in word:
			vector_list.append(vec)
	tensor = torch.Tensor(vector_list)
	mn = mean(tensor)
	std = variance(tensor) ** 0.5
	t = torch.zeros(tensor.shape)
	normalized = torch.addcdiv(t, (tensor - mn) , std)
	embedded= []
	for syll in normalized:
		embedded.append(mean_pad(syll, 1000).tolist())
	word_list = []
	size = 0
	for n in word_size:
		word_list.append(embedded[size: size + n])
		print('appended word' , len(word_list[-1]))
		size += n
	return word_list
		
	
def reindex(word_list: list, batch: str, i: int, data_dict: dict):
	'''
	word list is assumed to be arranged so that each syllable class comes in ascending order and within each class the words are sorted by ascending index. the relevant index values are obtained by looking at
	feature i of data_dict. the returned object is a dictionary which contains the same information keyed by the appropriate index.
	'''
	syllable_list = ['syllable_2', 'syllable_3', 'syllable_4']
	returned_object = {}
	n = 0
	for syllable in syllable_list:
		returned_object[syllable] = {}
		word_dict = data_dict[syllable][batch][i]
		keys = sorted(word_dict.keys())
		for key in keys:
			returned_object[syllable][key] = word_list[n]
			n +=1
	return returned_object
		
			
def write_data(data_dict: dict):
	'''
	data_dict contains all of the raw spectral features. This function collects the relevant batch for every feature, and normalizes over the whole set, embeds each into a size of 1000, then saves each of the batches as a list of vectors
	input shape: dict
	output shape: [ f1, f2, ..., fh] where fi = [ w1, ..., wm]
	'''
	syllable_list = ['syllable_2', 'syllable_3', 'syllable_4']
	batch_list = ['train', 'test', 'dev']
	number_of_features = len(data_dict['syllable_2']['train'].keys())
	features_train = {}
	features_test  = {}
	features_dev = {}
	for i in range(number_of_features):
		output_list = []
		batch_sizes = []
		for batch in batch_list:
			size = 0
			for syllable in syllable_list:
				word_dict = data_dict[syllable][batch][i]
				word_tuple = sorted(word_dict.items(), key = lambda x: x[0])
				word_list = [w[1] for w in word_tuple]
				output_list += word_list
				size += len(word_list)
			batch_sizes.append(size)
		print('batches found', batch_sizes)
		embedded_output = normalize_and_pad(output_list)
		train_size, test_size, dev_size = batch_sizes[0], batch_sizes[1], batch_sizes[-1]
		train_list, test_list, dev_list = embedded_output[:train_size], embedded_output[train_size: train_size + test_size], embedded_output[train_size + test_size:]
		train_list = reindex(train_list, 'train', i, data_dict)
		test_list = reindex(test_list, 'test', i, data_dict)
		dev_list = reindex(dev_list, 'dev', i, data_dict)
		print('train size', train_size)
		print('train features generated', len(train_list))
		print('test size', test_size)
		print('test features generated', len(test_list))
		print('dev size', dev_size)
		print('dev features generated', len(dev_list))
		features_train[i] = train_list
		features_test[i] = test_list
		features_dev[i] = dev_list
	subprocess.run(['rm', 'data/vectors/features/train/input.json'])
	path1 = Path('data/vectors/features/train/input.json')
	subprocess.run(['rm', 'data/vectors/features/test/input.json'])
	path2 = Path('data/vectors/features/test/input.json')
	subprocess.run(['rm', 'data/vectors/features/dev/input.json'])
	path3 = Path('data/vectors/features/dev/input.json')
	print('writing feature batches', path1, path2, path3)
	json.dump(features_train, path1.open(mode='w'))
	json.dump(features_test, path2.open(mode='w'))
	json.dump(features_dev, path3.open(mode='w'))
		
		
def balance_class(data: DataFrame):
	'''
	each data frame has locational imbalance, this needs to be addressed by oversampling the minority locations. 
	'''
	stress = data['stress']
	counts = {}
	for i in stress:
		if i in counts.keys():
			counts[i] += 1
		else:
			counts[i] = 1
	totals = []
	for key, value in counts.items():
		totals.append(value)
	print('class distribution', counts)
	majority = max(totals)
	remainders = []
	for i in counts.keys():
		idata = data[data['stress'] == i]
		print('oversampling ', majority - len(idata), 'from class', i)
		remainder = idata.sample(n = majority - len(idata), replace=True)
		remainders.append(remainder)
	remainders.append(data)
	value = pd.concat(remainders)
	print('balanced location')
	print(value.head(20))
	print(value.tail(20))
	print(value.index)
	return value
		
	
		

def balance_data():
	'''
	handles the balancing of training data. Train data is assumed to be classified into two schemes. The first scheme is within a single syllable type. We have minority classes within a single syllable type.
	These are oversampled. Secondly, there is the scheme over total syllable classes. There are more two syllable words than three and more three syllable words than four. Elements of three and four are also
	oversampled to match the number of elements in two. This is done only after each individual syllable type is balanced within its own class
	'''	
	print('balancing classes')
	data_2 = pd.read_csv('data/data_2/train/train.csv')
	data_2 = data_2.set_index('Unnamed: 0')
	print('data_2 read, indices', data_2.index)
	data_3 = pd.read_csv('data/data_3/train/train.csv')
	data_3 = data_3.set_index('Unnamed: 0')
	print('data_3 read, indices', data_3.index)
	data_4 = pd.read_csv('data/data_4/train/train.csv')
	data_4 = data_4.set_index('Unnamed: 0')
	print('data_4 read, indices', data_4.index)
	
	data_2 = balance_class(data_2)
	data_3 = balance_class(data_3)
	data_4 = balance_class(data_4)
	
	majority = max(len(data_2), len(data_3), len(data_4))
	print('balancing across every class, majority class has', majority)
	
	print('oversampling', majority - len(data_2), 'from data_2 with', len(data_2))
	remainder = data_2.sample(n = majority - len(data_2))
	subprocess.run(['rm', 'data/data_2/train/train_balanced.csv'])
	data_2 = pd.concat([data_2, remainder])
	data_2.to_csv('data/data_2/train/train_balanced.csv')
	
	print('oversampling', majority - len(data_3), 'from data_3 with', len(data_3))
	remainder = data_3.sample(n = majority - len(data_3))
	subprocess.run(['rm', 'data/data_3/train/train_balanced.csv'])
	data_3 = pd.concat([data_3, remainder])
	data_3.to_csv('data/data_3/train/train_balanced.csv')
	
	print('oversampling', majority - len(data_4), 'from data_4 with', len(data_4))
	remainder = data_4.sample(n = majority - len(data_4), replace=True)
	subprocess.run(['rm', 'data/data_4/train/train_balanced.csv'])
	data_4 = pd.concat([data_4, remainder])
	data_4.to_csv('data/data_4/train/train_balanced.csv')
	

def extract(data: int | pd.Series):
	'''
	sometimes an index is duplicated, so it returns a Series object instead of an integer. This function returns the value in that series object
	'''
	if type(data) == pd.Series:
		return list(data)[0]
	else:
		return data

def prepare_data(batch: str):
	'''
	train data is handled separately from test and dev. Since the data is highly imbalanced, training data is evened out, using simple duplication in order to oversample the minority classes.
	test and dev are processed as usual. All data is saved as json objects in data/train/input.json data/train/gold.json and data/batch/data_n for train and remaining respectively.
	'''
	print('preparing', batch)
	if batch == 'train':
		balance_data()
	input_list = []
	gold_list = []
	path_list = [f'data/data_2/{batch}',
				f'data/data_3/{batch}',
				f'data/data_4/{batch}'
				]
	total = 0
	meta = {}
	shutil.rmtree(f'data/vectors/{batch}/input')
	os.mkdir(f'data/vectors/{batch}/input')
	for n, path in enumerate(path_list):
		syl = n + 2
		meta[f'syllable_{syl}'] = {}
		meta[f'syllable_{syl}']['files'] = []
		csv_path = ''
		if batch == 'train':
			csv_path = Path(f'{path}/{batch}_balanced.csv')
		else:
			csv_path = Path(f'{path}/{batch}.csv')
		directory = Path(f'{path}/alignment/syllaudio')
		print('reading table from', csv_path)
		print('reading audio from', directory)
		print('current syllable class', syl)
		
		dataframe = pd.read_csv(csv_path)
		dataframe = dataframe.set_index('Unnamed: 0')
		sylcounter = 0
		for i in dataframe.index:
			word = []
			for j in range(syl):
				print('reading', f'file_{i}_syll{j+1}.wav')
				word.append(next(directory.glob(f'file_{i}_syll{j+1}.wav')))
			print('reading in word', word)
			padded_word = []
			for syllable in word:
				audio, samplerate = torchaudio.load(syllable)
				print('shape of audio signal', audio.shape)
				padded_audio = pad(audio[0], 30000)
				padded_word.append(padded_audio.tolist())
			input = Path(f'data/vectors/{batch}/input/{total}_{syl}_{i}.json')
			json.dump(padded_word, input.open(mode='w'))
			print('writing input to directory', input)
			gold_list.append([syl, i, binarize(extract(dataframe['stress'][i]), len(word))])
			meta[f'syllable_{syl}']['files'].append([total, syl, i])
			total += 1
			print('gold value obtained', gold_list[-1])
			sylcounter +=1
		meta[f'syllable_{syl}']['size'] = sylcounter
	path = Path(f'data/vectors/{batch}/input/meta.json')
	meta['size'] = total
	json.dump(meta, path.open(mode='w'))
	gold = Path(f'data/vectors/{batch}/gold.json')
	print('writing gold to directory', gold)
	json.dump(gold_list, gold.open(mode='w'))
	
	
if __name__ == '__main__':
	prepare_data('train')
	prepare_data('test')
	prepare_data('dev')
	
	data_dict = PrepareSpectralFeatures.prepare_data()
	write_data(data_dict)
			
				
			
			
		
	
		
		
	
	