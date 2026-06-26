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
	optim = torch.optim.SGD(network.parameters(), lr=0.001,  momentum=.5)
	path = Path(f'{input}/meta.json')
	meta = json.load(path.open(mode='r'))
	N1 = meta['syllable_2']['size']
	N2 = meta['syllable_3']['size']
	N3 = meta['syllable_4']['size'] 
	#print('randomly sampling from', 0, 'to', N1+ N2 + N3)
	#sample_analysis(0, N1 + N2 + N3, input, gold)
	#print(meta.keys())
	#print('syllable 2 size', meta['syllable_2']['size'])
	#print('syllable 3 size', meta['syllable_3']['size'])
	#print('syllable 4 size', meta['syllable_4']['size'])
	epoch = 220
	network.feature_weights = []
	while epoch > 0:
		optim.zero_grad()
		error = 0
		n = torch.randint(0, N1 + N2 + N3, (25,))
		#n = [0, N1, N1 + N2]
		for i in n:
			y = torch.tensor(gold[i][-1])
			syll = gold[i][0]
			index = gold[i][1]
			xpath = Path(f'{input}/{i}_{syll}_{index}.json')
			print('reading file', xpath)
			x = torch.tensor(json.load(xpath.open(mode='r'))) 
			print('input vector statistics', 'length', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), 'dimensions', torch.count_nonzero(x, dim=1).tolist())
			f = torch.tensor(filter(features, syll, index))
			print('input feature statistics', f.shape, torch.min(f).item(), torch.max(f).item(), torch.mean(x).item())
			y_hat = network.forward(x, f)
			#print('forward pass complete')
			print( y_hat, 'gold', y)
			error += compute_loss(y_hat, y)
		print('batch error', error)
		error_history.append(error.item())
		#print("error analysis before backpropagation")
		#analyze_graph(error.grad_fn)
		print('epoch', epoch)
		print('backpropagating the error')
		#print(optim.state_dict())
		error.backward()
		#analyze_state_dict(network.state_dict())
		#print(optim.state_dict())
		print('updating the parameters')
		optim.step()
		#analyze_optimstate_dict(optim.state_dict())
		epoch -= 1
	network.cycles +=1
	print('training complete. writing network parameters to directory.')
	state_dict = network.state_dict(keep_vars = True)
	for key, value in state_dict.items():
		state_dict[key] = value.tolist()
	path = Path('neural_network/model.json')
	path.touch()
	json.dump(state_dict, path.open(mode='w'))
	print('error history', error_history)
	print(torch.min(torch.tensor(error_history)).item(), torch.max(torch.tensor(error_history)).item(), torch.mean(torch.tensor(error_history)).item())


def analyze_optimstate_dict(state_dict: dict):
	'''
	looks at model's previous gradients for any anomalies.
	'''
	for key, value in state_dict['state'].items():
		gradient = value['momentum_buffer']
		print(key, torch.min(gradient), torch.max(gradient), gradient.shape)
	
	
def analyze_state_dict(state_dict: dict):
	'''
	looks at the models parameters for any anomalies.
	'''
	for key, value in state_dict.items():
		print(key, value.shape)
	
	
	
def sample_analysis(N1: int, N: int, input:str, gold:list):
	'''
	randomly samples classes and analyzes the distribution
	'''
	n = torch.randint(N1, N, (1000,))
	d = {1:0, 2:0, 3:0, 4:0}
	k = {2:0, 3:0, 4:0}
	for i in n:
		value = gold[i]
		syll = value[0]
		k[syll] +=1
		location = value[-1]
		for j in range(syll):
			if location[j] == 1:
				d[j+1] +=1
	distribution = []
	distributionw = []
	for key, value in d.items():	
		distribution.append(value / 1000)
	for key, value in k.items():
		distributionw.append(value / 1000)
	print('distribution obtained over location classes')
	print(distribution)
	print('distribution obtained over syllable classes')
	print(distributionw)
				
		
	
	
def analyze_graph(y: Tensor):
	'''
	analyzes the graph function of y
	'''
	if y == None:
		return
	print(y.next_functions)
	for f, k in y.next_functions:
		analyze_graph(f)
	
	
def filter(features:dict, syll: int, i:int):
	'''
	each vector can be identified by its batch (which in this case is assumed to be train), syllable class, and index 
	'''
	f = []
	for feature in features.keys():
		f.append(features[f'{feature}'][f'syllable_{syll}'][f'{i}'])
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
	#print('distance to be computed between')
	#print(y_hat)
	#print(y)
	distance = torch.add(y, y_hat, alpha=-1)
	#print(distance, torch.linalg.vecdot(distance, distance))
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
	input = 'data/vectors/train/input'
	gold_path = Path('data/vectors/train/gold.json')
	feature_path = Path('data/vectors/features/train/input.json')
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
	print('input', input)
	print('gold shape', len(gold))
	train(network, input, gold, features)
	
	