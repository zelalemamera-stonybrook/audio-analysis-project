'''
this program controls the training of the StressClassifier neural network. our inputs are assumed to be in the required form. we load in the data and optimize the network using SGD algorithm. 
'''
import StressClassifier
from StressClassifier import Network
import argparse
from pathlib import Path
import json
import torch
from torch import Tensor


def train(network: Network , input: list[Tensor], gold: list[Tensor], features: list[list[Tensor]]):
	'''
	takes in the network, the input audio data and the targets, optimizes the model that best represents this target over the input data.
	input shape: ( n, 30,000), where n is the number of syllables  for each sample
			: (n, 2)
	output shape: none
	
	'''
	optim = torch.optim.SGD(network.parameters(), lr=0.001,  momentum=1)
	N = len(input)
	epoch = 100
	while epoch > 0:
		optim.zero_grad()
		error = 0
		n = torch.randint(N, (20,))
		for i in n:
			x, y = input[i], gold[i]
			f = filter(features, i)
			y_hat = network.forward(x, f)
			error += compute_loss(y_hat, y)
		error.backward()
		optim.step()
		epoch -= 1
	network.cycles +=1
	state_dict = network.state_dict(keep_vars = True)
	for key, value in state_dict.items():
		state_dict[key] = value.tolist()
	path = Path('model.json')
	path.touch()
	json.dump(state_dict, path.open(mode='w'))
	
def filter(features:list[list[Tensor]], i: int):
	'''
	returns the ith tensor of each list in features
	output: list[Tensor]
	'''
	f = []
	for word_list in features:
		f.append(word_list[i])
	return f
		
def compute_loss(y_hat: Tensor, y: Tensor):
	'''
	measures the distance of the prediction y hat to y and returns the result.
	y_hat shape: (n, 2)
	y shape: (n)
	'''
	binary_list = []
	for n in y:
		binary_list.append(binarize(n, 2))
	y = torch.Tensor(binary_list)
	distance = torch.add(y, y_hat, alpha=-1)
	error = 0.5 * (torch.linalg.vecdot(distance, distance)).sum()
	return error
	
def binarize(i: int, n: int):
	'''
	returns a list of length n, with all zeros except at position i
	'''
	output = torch.zeros((n,))
	output[ i ] = 1
	return output.tolist()
	
def read_in_data():
	'''
	reads in the training input and gold data from the current directory, also reads in features
	'''
	input_path = Path('train_data/input.json')
	gold_path = Path('train_data/gold.json')
	feature_path = Path('train_features.json')
	input = json.load(input_path.open(mode='r'))
	gold = json.load(gold_path.open(mode='r'))
	features = json.load(feature_path.open(mode='r'))
	
	input_tensor = []
	gold_tensor = []
	features_tensor= []
	
	for word in input:
		input_tensor.append(torch.Tensor(word))
	for word in gold:
		gold_tensor.append(torch.Tensor(word))
	for f in features:
		word_list = []
		for word in f:
			word_list.append(torch.Tensor(word))
		features_tensor.append(word_list)
	return input_tensor, gold_tensor, features_tensor
			
			
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--reset', action = 'store_true', help='reset model parameters')
	args = parser.parse_args()
	
	network = None
	if args.reset:
		network = StressClassifier.Network()
	else:
		path = Path('model.json')
		model_parameters = json.load(path.open(mode='r'))
		for key, value in model_parameters.items():
			model_parameters[key] = torch.Tensor(value)
		network = StressClassifier.Network(model_parameters)
	print('reading in training data')
	input, gold, features = read_in_data()
	print('input shape', len(input), input[-1].shape)
	print('gold shape', len(gold), gold[-1].shape)
	print('features shape', len(features), len(features[-1]), features[-1][-1].shape)
	train(network, input, gold, features)
	
	