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


def train(network: Network , input: list, gold: list, features: dict):
	'''
	takes in the network, the input audio data and the targets, optimizes the model that best represents this target over the input data.
	input shape: ( n, 30,000), where n is the number of syllables  for each sample
			: (n, 2)
	output shape: none
	
	'''
	error_history = []
	optim = torch.optim.SGD(network.parameters(), lr=0.001,  momentum=1)
	N = len(input)
	epoch = 100
	while epoch > 0:
		optim.zero_grad()
		error = 0
		n = torch.randint(N, (20,))
		for i in n:
			x, y = torch.tensor(input[i][-1]), torch.tensor(gold[i][-1])
			syll = input[i][0]
			index = input[i][1]
			f = torch.tensor(filter(features, syll, index))
			print(f'passing random word {i} to the network', x.shape, y.shape, f.shape)
			y_hat = network.forward(x, f)
			print('forward pass complete')
			error += compute_loss(y_hat, y)
		print('batch error', error)
		error_history.append(error.item())
		print('backpropagating the error')
		error.backward()
		print('updating the parameters')
		optim.step()
		epoch -= 1
	network.cycles +=1
	print('training complete. writing network parameters to directory.')
	state_dict = network.state_dict(keep_vars = True)
	for key, value in state_dict.items():
		state_dict[key] = value.tolist()
	path = Path('model.json')
	path.touch()
	json.dump(state_dict, path.open(mode='w'))
	print('error history', error_history)
	
def filter(features:dict, syll: int, i:int):
	'''
	each vector can be identified by its batch (which in this case is assumed to be train), syllable class, and index 
	'''
	f = []
	for feature in features.keys():
		f.append(features[feature][f'syllable_{syll}'][i])
	return f
		
def compute_loss(y_hat: Tensor, y: Tensor):
	'''
	measures the norm of distance of the prediction y hat to y and returns the result.
	y_hat shape: (n, 2)
	y shape: (n)
	'''
	binary_list = []
	for n in y:
		binary_list.append(binarize(int(n.item()), 2))
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
	input_path = Path('data/vectors/train/input.json')
	gold_path = Path('data/vectors/train/gold.json')
	feature_path = Path('data/vectors/features/train/input.json')
	input = json.load(input_path.open(mode='r'))
	gold = json.load(gold_path.open(mode='r'))
	features = json.load(feature_path.open(mode='r'))
	
	return input, gold, features
			
			
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--reset', action = 'store_true', help='reset model parameters')
	args = parser.parse_args()
	
	network = None
	if args.reset:
		network = StressClassifier.Network()
	else:
		path = Path('neural_network/model.json')
		model_parameters = json.load(path.open(mode='r'))
		for key, value in model_parameters.items():
			if key == 'cycles':
				model_parameters[key] = torch.tensor(value)
			model_parameters[key] = torch.nn.parameter.Parameter(torch.tensor(value), requires_grad = True)
		network = StressClassifier.Network()
		network.load_state_dict(model_parameters)
	print('reading in training data')
	input, gold, features = read_in_data()
	print('input shape', len(input), input[:10])
	print('gold shape', len(gold) gold[:10])
	print('features shape', features.keys(), features[0].keys(),features[0]['syllable_4'].keys())
	train(network, input, gold, features)
	
	