'''
the following program tests the performance of StressClassifier neural network. We generate f score, accuracy, precision, and recall over multiple syllables.
in addition, this program also presents the weights that the neural network learned for feature embeddings. we use this to make a judgement about which features are important.
'''
import json
from pathlib import Path
import StressClassifier
import torch
from torch import Tensor
import pandas as pd

def test(network: StressClassifier, input: list, gold: list, features: dict):
	'''
	this is the main function of the program. we pass the input words once through the network, and evaluate the results.
	'''
	
	#size_list  = count_syllable_sizes(input)
	#two_index = [i for i in range(size_list[0])]
	#three_index = [ i + size_list[0] for i in range(size_list[1])]
	#four_index = [i + size_list[0] + size_list[1] for i in range(size_list[-1])]
	#two_data, three_data, four_data = split_by_syllable(input, gold, features, size_list)
	evaluate(network, input, 2, gold)
	evaluate(network, input, 3, gold)
	evaluate(network, input, 4, gold)
	
	
def evaluate(network: StressClassifier, input: str, syll: int, gold: list):
	'''
	passes, the input data through the network and uses the output to write an evaluation for the model performance
	'''
	print('evaluating model performance')
	#input, golds, features = input_data
	output = []
	gold_list = []
	path = Path(f'{input}/meta.json')
	meta = json.load(path.open(mode='r'))
	files = meta[f'syllable_{syll}']['files']
	network.feature_weights = []
	for file in files:
		print('received file', file)
		i = file[-1]
		n = file[0]
		print('reading in gold value', gold[n])
		gold_list.append(torch.tensor(gold[n][-1]))
		f = torch.Tensor(filter(features, syll, i))
		xpath = Path(f'{input}/{file[0]}_{file[1]}_{file[-1]}.json')
		x = torch.Tensor(json.load(xpath.open(mode='r')))
		print(f'sequentially passing word {i} to the network', x.shape, f.shape)
		y_hat = network.forward(x, f)
		output.append(y_hat.detach())
	cycles = network.cycles.item()
	r = recall(output, gold_list)
	p = precision(output, gold_list)
	f = fscore(r, p)
	write_results(cycles, syll, r, p, f, 'dev')
	write_feature_weights(output, gold_list, network.feature_weights, 'dev', input)
	
def write_results(cycles:int, syllables: int, r: float, p:float, fs: float, batch: str):
	'''
	writes the result to the directory 
	'''
	print('writing result table')
	path = Path(f'results/syllable_{syllables}/{batch}/model_performance.txt')
	if path.exists():
		with path.open(mode='a') as f:
			line = f'{cycles}\t{syllables}\t{r}\t{p}\t{fs}\n'
			f.write(line)
	else:
		path.touch()
		with path.open(mode='w') as f:
			line = f'cycles\tsyllables\trecall\tprecision\tfscore\n'
			f.write(line)
			line = f'{cycles}\t{syllables}\t{r}\t{p}\t{fs}\n'
			f.write(line)
	
def write_feature_weights(output: list[Tensor], gold: list[Tensor], attention_weights: list[list], batch: str, input:str):
	'''
	writes a file containing the model's performance on each syllable and the weights that it assigned to each syllable.
	'''
	print('writing feature analysis')
	syllable = len(gold[0])
	path = Path(f'{input}/meta.json')
	print('reading meta data from', path)
	meta = json.load(path.open(mode='r'))
	files = meta[f'syllable_{syllable}']['files']
	print('files to be read', files)
	path = Path(f'results/syllable_{syllable}/{batch}/feature_analysis.txt')
	print('writing to file', path)
	table_path = Path(f'data/data_{syllable}/{batch}/{batch}.csv')
	print('reading from table', table_path)
	table = pd.read_csv(table_path)
	table = table.set_index('Unnamed: 0')
	with path.open('w') as f:
		line = f'ipa\tsyllable\tprobability\tpredicted\tgold\tcorrect\tfeatures\n'
		f.write(line)
		for i, file in enumerate(files):
			index = file[-1]
			ipa = table['ipa'][index]
			for j in range(syllable):
				if j == 0:
					line = f'{ipa}\t{j + 1}\t{to_list(output[i][j])}\t{torch.argmax(output[i][j])}\t{gold[i][j]}\t{ gold[i][j] == torch.argmax(output[i][j])}\t{[round(n, 3) for n in attention_weights[i][j]]}\n'
					f.write(line)
				else:
					line = f'\t{j + 1}\t{to_list(output[i][j])}\t{torch.argmax(output[i][j])}\t{gold[i][j]}\t{ gold[i][j] == torch.argmax(output[i][j])}\t{[round(n, 3) for n in attention_weights[i][j]]}\n'
					f.write(line)
					
def to_list(t: Tensor):
	'''
	converts this tensor to a list in a specific format that is desired for the output text file.
	'''
	
	rounded = [round(v.item(), 2) for v in t]
	return rounded
	
def recall(output: list[Tensor],  gold: list):
	'''
	computes the recall of output against gold.
	'''
	y_hat = []
	for word in output:
		classification_value = []
		for distribution in word:
			classification_value.append(torch.argmax(distribution))
		y_hat.append(classification_value)
	recall = 0
	for prediction, true in zip(y_hat, gold):
		i = torch.argmax(torch.tensor(true))
		if prediction[i] == 1:
			recall +=1
	return round(recall / len(gold), 2)
   
def precision(output: list[Tensor],  gold: list):
	'''
	computes the precision as the proportion of true positives to all positives
	'''
	y_hat = []
	for word in output:
		classification_value = []
		for distribution in word:
			classification_value.append(torch.argmax(distribution))
			y_hat.append(classification_value)
	precision = 0
	total = 0
	for prediction, true in zip(y_hat, gold):
		for i, val in enumerate(prediction):
			if val == 1:
				total +=1
				if true[i] == 1:
					precision+=1
	if total == 0:
		return 0
	return round(precision/total, 2)
	
  
def fscore(recall: float, precision: float):
	'''
	computes the fscore measure
	'''
	if recall + precision == 0:
		return None
	f = round(2 * (recall * precision)/ (recall + precision), 2)
	return f
		
def filter(features: dict, syll: int, i: int):
	'''
	returns the ith element of each list in features
	'''
	f = []
	for key, value in features.items():
		f.append(value[f'syllable_{syll}'][f'{i}'])
	return f
		
def split_by_syllable(input: list, gold: list, features: list, size_list: list[int]):
	'''
	splits the input batch by syllables, each batch is always sorted by syllable size, size list contains the number of words in each batch. The result is returned as a triple.
	'''
	two = size_list[0]
	three = size_list[1]
	four = size_list[-1]
	
	input = sorted(input, key = lambda x: (x[0], x[1]))
	gold = sorted(gold, key = lambda x: (x[0], x[1]))
	
	two_input, two_gold, two_features = input[:two], gold[:two], [features[f]['syllable_2'] for f in features.keys()]
	three_input, three_gold, three_features = input[two:two + three], gold[two:two + three], [features[f]['syllable_3'] for f in features.keys()]
	four_input, four_gold, four_features = input[two + three:], gold[two + three:], [features[f]['syllable_4'] for f in features.keys()]
	
	return (two_input, two_gold, two_features), (three_input, three_gold, three_features), (four_input, four_gold, four_features)
	
	
def count_syllable_sizes(input: list):
	'''
	the input contains a list of words that are already sorted by syllable size. 
	we count how many of them are there in each and return the result as a list
	'''
	size_list = []
	path = Path(f'{input}/meta.json')
	meta = json.load(path.open(mode='r'))
	two_size = meta['syllable_2']
	three_size = meta['syllable_3']
	four_size = meta['syllable_4']
	size_list.append(two_size)
	size_list.append(three_size)
	size_list.append(four_size)
	return size_list
			


def read_in_data(batch: str):
	'''
	reads in input and gold data from the current directory, also reads in features
	'''
	print('reading in testing data')
	input_path, gold_path, feature_path = Path(f'data/vectors/{batch}/input'), Path(f'data/vectors/{batch}/gold.json'), Path(f'data/vectors/features/{batch}/input.json')
	input, gold, features = input_path, json.load(gold_path.open(mode='r')), json.load(feature_path.open(mode='r'))

	return input, gold, features

def load_model():
	'''
	loads in the trained model parameters and initializes the network for testing. 
	'''
	print('loading in model')
	path = Path('neural_network/model.json')
	state_dict = json.load(path.open(mode='r'))
	for key, value in state_dict.items():
		if key == 'cycles':
			state_dict[key] = torch.tensor(value)
		state_dict[key] = torch.nn.parameter.Parameter(torch.tensor(value), requires_grad = True)
	network = StressClassifier.Network()
	network.load_state_dict(state_dict)
	return network

if __name__== '__main__':
	network = load_model()
	input, gold, features = read_in_data('dev')
	test(network, input, gold, features)
	
	
	