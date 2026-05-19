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


def train(network: Network , input: list[Tensor], gold: list[Tensor], features: list[Tensor]):
	'''
	takes in the network, the input audio data and the targets, optimizes the model that best represents this target over the input data.
	input shape: ( n, 10,000), where n is the number of syllables  for each sample
			: (n, 2)
	output shape: none
	
	'''
	optim = torch.optim.SGD(network.parameters(), lr=0.001,  momentum=1)
	N = len(input)
	epoch = 100
	while epoch > 0:
		optim.zero_grad()
		error = 0
		n = torch.randint(0, N, 20)
		for i in n:
			x, y = input[i], gold[i]
			y_hat = network.forward(x)
			error += compute_loss(y_hat, y)
		error.backward()
		optim.step()
		epoch -= 1
		
def compute_loss(y_hat: Tensor, y: Tensor)
	'''
	measures the distance of the prediction y hat to y and returns the result.
	y_hat shape: (n, 2)
	y shape: (n, 2)
	'''
	distance = torch.add(y, y_hat, alpha=-1)
	error = 0.5 * (torch.linalg.vecdot(distance, distance)).sum()
	return error
	
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
	
	for v in input:
		
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(''-r', '--reset', action = 'store_true', help='reset model parameters from past training')
	args = parser.parse_args()
	
	network = none
	if args.reset:
		network = StressClassifier.Network()
	else:
		path = Path('model')
		model_parameters = json.load(path.open(mode='r'))
		for key, value in model_parameters.items():
			model_parameters[key] = torch.Tensor(value)
		network = StressClassifier.Network(model_parameters)
	input, gold, features = read_in_data()
	train(network, input, gold, features)
	
	