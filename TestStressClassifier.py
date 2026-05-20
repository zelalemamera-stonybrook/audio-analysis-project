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

def test(network: StressClassifier, input: list[Tensor], gold: list[Tensor], features: list[list[Tensor]])
	'''
	this is the main function of the program. we pass the input words once through the network, and evaluate the results.
	'''
	
	size_list  = count_syllable_sizes(input)
	two_data, three_data, four_data = split_by_syllable(input, gold, features, size_list)
	evaluate(network, two_data)
	evaluate(network, three_data)
	evaluate(network, four_data)
	
	
def evaluate(netwok: StressClassifier, input_data: tuple):
	'''
	passes, the input data through the network and uses the output to write an evaluation for the model performance
	'''
	input, gold, features = input_data
	output = []
	attention_weights = []
	for i, word in enumerate(input):
		f = filter(features, i)
		y_hat = network.forward(word, f)
		output.append(y_hat)
		attention_weights.append(network.attention_weights)
	
	cycles, syllables, recall, precision, fscore = network.cycles.item(), len(gold[0]), recall(output, gold), precision(output, gold), f_score(recall(output, gold), precision(output, gold))
	write_results(cycles, syllables, acc, precision, fscore, 'dev')
	write_feature_weights(output, gold, attention_weights, 'dev')
	
def write_results(cycles:int, syllables: int, acc: float, precision:float, fscore: float, batch: str):
	'''
	writes the result to the directory 
	'''
	path = (f'results/syllable_{syllables}/{batch}/model_performance.txt')
	if path.exists()
		with path.open(mode='a') as f:
			line = f'{cycles}\t{syllables}\t{recall}\t{precision}\t{fscore}\n'
			f.write(line)
	else:
		path.touch()
		with path.open(mode='w') as f:
			line = f'cycles\tsyllables\trecall\tprecision\tfscore\n'
			f.write(line)
			line = f'{cycles}\t{syllables}\t{recall}\t{precision}\{fscore}\n'
			f.write(line)
	
def write_feature_weights(output: list[Tensor], gold: list[Tensor], attention_weights: list[list], batch: str):
	'''
	writes a file containing the model's performance on each syllable and the weights that it assigned to each syllable.
	'''
	syllable = len(gold[0])
	path = Path(f'results/syllable_{syllable}/{batch}/feature_analysis.txt')
	table_path = next((Path(f'{batch}_data/syllable_{syllable}')).glob('**/*.csv))
	table = pd.read_csv(table_path)
	ipa = table['ipa']
	with path.open('w') as f:
		line = f'ipa\tsyllable\tprobability\tgold\tcorrect\tfeatures\n'
		f.write(line)
		for i, ipa in enumerate(ipa):
			for j in range(syllable):
				if j == 0:
					line = f'{ipa}\t{j + 1}\t{torch.round(output[i][j], decimals=2)}\t{gold[i][j]}\t{ gold[i][j] == torch.argmax(output[i][j])}\t{attention_weights[i][j]}\n
					f.write(line)
				else:
					line = f'\t\t{j + 1}\t{torch.round(output[i][j], decimals=2)}\t{gold[i][j]}\t{ gold[i][j] == torch.argmax(output[i][j])}\t{attention_weights[i][j]}\n
					f.write(line)
					
def recall(output: list[Tensor],  gold: list[Tensor]):
   '''
   computes the recall of output against gold.
   '''
   y_hat = []
   for word in output:
   	classification_value = []
   	for distribution in word:
		classification_value.append(torch.argmax(distribution))
	y_hat.append(torch.Tensor(classification_value))
   recall = 0
   for prediction, true in zip(y_hat, gold):
   	i = torch.argmax(true)
	if prediction[i] == 1:
		recall +=1
   return round(recall / len(gold), 2)
   
def precision(output: list[Tensor],  gold: list[Tensor]):
    '''
    computes the precision as the proportion of true positives to all positives
    '''
    y_hat = []
   for word in output:
   	classification_value = []
   	for distribution in word:
		classification_value.append(torch.argmax(distribution))
	y_hat.append(torch.Tensor(classification_value))
   precision = 0
   total = 0
   for prediction, true in zip(y_hat, gold):
   	for i, val in enumerate(prediction):
		if val == 1:
			total +=1
			if true[i] == 1:
				precision+=1
   return round(precision/total, 2)
  
  def fscore(recall: float, precision: float):
  	'''
	computes the fscore measure
	'''
	f = round(2 * (recall * precision)/ (recall + precision), 2)
		
def filter(features:list[list[Tensor]], i: int):
	'''
	returns the ith tensor of each list in features
	'''
	f = []
	for feature in features:
		f.append(feature[i])
	return f
		
def split_by_syllable(input: list[Tensor], gold: list[Tensor], features: list[list[Tensor]], size_list: list[int])
	'''
	splits the input batch by syllables, each batch is always sorted by syllable size, size list contains the number of words in each batch. The result is returned as a triple.
	'''
	two = size_list[0]
	three = size_list[1]
	four = size_list[-1]
	
	two_input, two_gold, two_features = input[:two], gold[:two], [f[:two] for f in features]
	three_input, three_gold, three_features = input[two:two + three], gold[two:two + three], [f[two:two + three] for f in features]
	four_input, four_gold, four_features = input[two + three:], gold[two + three:], [f[two + three:] for f in features]
	
	return (two_input, two_gold, two_features), (three_input, three_gold, three_features), (four_input, four_gold, four_features)
	
	
def count_syllable_sizes(input: list[Tensor])
	'''
	the input contains a list of words that are already sorted by syllable size. 
	we count how many of them are there in each and return the result as a list
	'''
	size_list = []
	n = 0
	s = 2
	for word in input:
		if  len(word) == s:
			n+=1
		else:
			size_list.append(n)
			n = 1
			s +=1
	return size_list
			


def read_in_data(batch: str):
	'''
	reads in input and gold data from the current directory, also reads in features
	'''
	input_path, gold_path, feature_path = Path(f'{batch}_data/input.json'), Path(f'{batch}_data/gold.json'), Path(f'{batch}_features.json')
	input, gold, features = json.load(input_path.open(mode='r')), json.load(gold_path.open(mode='r')), json.load(feature_path.open(mode='r'))
	
	input_tensor = []
	gold_tensor = []
	features_tensor= []
	
	for word in input:
		input_tensor.append(torch.Tensor(word)
	for word in gold:
		gold_tensor.append(torch.Tensor(word))
	for f in features:
		word_list = []
		for word in f:
			word_list.append(torch.Tensor(word)
		features_tensor.append(word_list)
	return input_tensor, gold_tensor, features_tensor

def load_model():
	'''
	loads in the trained model parameters and initializes the network for testing. 
	'''
	path = Path('model.json')
	state_dict = json.load(path.open(mode='r'))
	for key, value in state_dict.items():
		state_dict[key] = torch.Tensor(value)
	network = StressClassifier.Network(state_dict)
	return network

if __name__== '__main__':
	network = load_model()
	input, gold, features = read_in_data('dev')
	test(network, input, gold, features)
	
	
	