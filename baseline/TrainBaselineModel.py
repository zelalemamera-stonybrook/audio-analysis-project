'''
This code tests the baseline model. One cycle is defined as the minimum number of epochs needed to see the entire dataset for the given batch size.
'''
import BaselineModel
from BaselineModel import Network
import argparse
from pathlib import Path
import json
import torch
import torchaudio
from torch import Tensor

def train_baseline(model, network: Network, input:str, gold:list):
	'''
	trains the baseline model one cycle and saves its weights in the directory
	'''
	error_history = []
	optim = torch.optim.SGD(network.parameters(), lr=0.001,  momentum=.5)
	path = Path(f'{input}/meta.json')
	meta = json.load(path.open(mode='r'))
	N1 = meta['syllable_2']['size']
	N2 = meta['syllable_3']['size']
	N3 = meta['syllable_4']['size'] 
	#print('randomly sampling from', 0, 'to', N1+ N2 + N3)
	#sample_analysis(0, N1, input, gold)
	#print(meta.keys())
	#print('syllable 2 size', meta['syllable_2']['size'])
	#print('syllable 3 size', meta['syllable_3']['size'])
	#print('syllable 4 size', meta['syllable_4']['size'])
	
	epoch = 55
	while epoch > 0:
		optim.zero_grad()
		error = 0
		n = torch.randint(0, N1 + N2 + N3, (25,))
		#n = [0, N1 + N2]
		for i in n:
			y = torch.tensor(gold[i][-1])
			syll = gold[i][0]
			index = gold[i][1]
			xpath = Path(f'{input}/{i}_{syll}_{index}.json')
			print('reading file', xpath)
			waveform = torch.tensor(json.load(xpath.open(mode='r'))) 
			vecs = []
			model.eval()
			with torch.no_grad():
				vecs, _ = model.extract_features(waveform)
			x = []
			for vec in vecs[-1]:
				x.append(vec.reshape(-1))
			x = torch.stack(x).detach()
			
			print('input vector statistics', 'length', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), 'dimensions', torch.count_nonzero(x, dim=1).tolist())	
			y_hat = network.forward(x)
			#print('forward pass complete')
			print( y_hat, 'gold', y)
			error += compute_loss(y_hat, y)
		print('batch error', error)
		error_history.append(error.item())
		print('epoch', epoch)
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
	path = Path('baseline/model.json')
	path.touch()
	json.dump(state_dict, path.open(mode='w'))
	print('error history', error_history)
	print(torch.min(torch.tensor(error_history)).item(), torch.max(torch.tensor(error_history)).item(), torch.mean(torch.tensor(error_history)).item())
		


def binarize(i: int, n: int):
	'''
	returns a list of length n, with all zeros except at position i
	'''
	output = torch.zeros((n,))
	output[ i ] = 1
	return output.tolist()
	
			
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

def read_in_data_train():
	'''
	reads in the training input and gold data from the current directory, also reads in features
	'''
	input = 'data/vectors/train/input'
	gold_path = Path('data/vectors/train/gold.json')
	gold = json.load(gold_path.open(mode='r'))
	
	return input, gold


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--reset', action = 'store_true', help='reset model parameters')
	args = parser.parse_args()
	
	network = None
	if args.reset:
		network = Network()
	else:
		path = Path('baseline/model.json')
		model_parameters = json.load(path.open(mode='r'))
		for key, value in model_parameters.items():
			if key == 'cycles':
				model_parameters[key] = torch.tensor(value)
			model_parameters[key] = torch.nn.parameter.Parameter(torch.tensor(value), requires_grad = True)
		network = BaselineModel.Network()
		network.load_state_dict(model_parameters)
	bundle = torchaudio.pipelines.WAV2VEC2_BASE
	model = bundle.get_model()
	print('reading in training data')
	input, gold = read_in_data_train()
	print('input', input)
	print('gold shape', len(gold))
	train_baseline(model, network, input, gold)
	
	
	
	
	
	
	
	