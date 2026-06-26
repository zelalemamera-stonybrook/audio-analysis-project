'''
This program implements the baseline against which our neural network is to be compared. The baseline is a simple MLP which takes as its input, a word, treated as wavtovec embedded syllables, and outputs a sequence of probability distributions.
'''

import torch
import torchaudio
import json
from pathlib import Path
import torch.nn as nn
from torch import Tensor
import argparse
import pandas as pd

class Network(nn.Module):
	'''
	neural network implementation of the above
	'''
	def __init__(self):
		super().__init__()
		
		self.cycles = nn.parameter.Parameter(torch.tensor(float(0)), requires_grad = False) 
		
		self.conv1 = nn.parameter.Parameter((torch.rand((9,)) - 0.5) * 9, requires_grad = True) # stride = 5
		self.conv1_bias = nn.parameter.Parameter(torch.rand((1,)), requires_grad = True)
		
		self.conv2 = nn.parameter.Parameter((torch.rand((5,)) - 0.5) * 5, requires_grad = True) # stride = 2
		self.conv2_bias = nn.parameter.Parameter(torch.rand((1,)), requires_grad = True)
		
		self.conv3 = nn.parameter.Parameter((torch.rand((6,)) -0.5) * 6, requires_grad = True) # stride = 2
		self.conv3_bias = nn.parameter.Parameter(torch.rand((1,)), requires_grad = True)
		
		self.conv4 = nn.parameter.Parameter((torch.rand((4,)) -0.5) * 4, requires_grad = True) # stride = 2
		self.conv4_bias = nn.parameter.Parameter(torch.rand((1,)), requires_grad = True)
		
		self.conv5 = nn.parameter.Parameter((torch.rand((4,)) - 0.5) * 4, requires_grad = True) # stride = 2
		self.conv5_bias = nn.parameter.Parameter(torch.rand((1,)), requires_grad = True) 
		
		self.conv6 = nn.parameter.Parameter((torch.rand((4,)) - 0.5) * 4, requires_grad = True) # stride = 2
		self.conv6_bias = nn.parameter.Parameter(torch.rand((1,)), requires_grad = True)
		
		
		self.left_rnn_in = nn.parameter.Parameter(torch.rand((500,444)) - 0.5, requires_grad = True)
		self.left_rnn_in_bias = nn.parameter.Parameter(torch.rand((500,)), requires_grad = True)
		self.left_rnn_hidden = nn.parameter.Parameter(torch.rand((500,500)) - 0.5, requires_grad = True)
		self.left_rnn_hidden_bias = nn.parameter.Parameter(torch.rand((500,)), requires_grad = True)
	
		self.right_rnn_in = nn.parameter.Parameter(torch.rand((500,444)) - 0.5, requires_grad = True)
		self.right_rnn_in_bias = nn.parameter.Parameter(torch.rand((500,)), requires_grad = True)
		self.right_rnn_hidden = nn.parameter.Parameter(torch.rand((500,500)) - 0.5, requires_grad = True)
		self.right_rnn_hidden_bias = nn.parameter.Parameter(torch.rand((500,)), requires_grad = True)
	
		self.output1 = nn.parameter.Parameter(torch.rand((500, 1000)) - 0.5, requires_grad = True)
		self.output1_bias = nn.parameter.Parameter(torch.rand((500,)), requires_grad = True)
		
		self.output2 = nn.parameter.Parameter(torch.rand((2, 500)) - 0.5, requires_grad = True)
		self.output2_bias = nn.parameter.Parameter(torch.rand((2,)), requires_grad = True)
		
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim = 0)
	
	def forward(self, input: list[Tensor]):
		'''
		the input is a list of vectors, these are first passed each to the MLP which converts them to 500, then the list is processed by a bidirectional rnn that outputs the probability distribution for each syllable
		'''
		output_list = []
		for vec in input:
			output_list.append(self.convolution_forward(vec))
		output = self.rnn_forward(output_list)
		return output
		
	def convolution_forward(self, vec: Tensor):
		'''
		passes vec through two convolution layers as specified and returns the result
		'''
		conv_first = self.tanh(self.convolve(self.conv1, vec, self.conv1_bias, stride=5))
		#print(conv_first.shape, torch.min(conv_first).item(), torch.max(conv_first).item(), torch.mean(conv_first).item())
		conv_second = self.tanh(self.convolve(self.conv2, conv_first,  self.conv2_bias, stride=2))
		#print(conv_second.shape, torch.min(conv_second).item(), torch.max(conv_second).item(), torch.mean(conv_second).item())
		conv_third = self.tanh(self.convolve(self.conv3, conv_second,  self.conv3_bias, stride=2))
		#print(conv_third.shape, torch.min(conv_third).item(), torch.max(conv_third).item(), torch.mean(conv_third).item())
		conv_fourth = self.tanh(self.convolve(self.conv4, conv_third,  self.conv4_bias, stride=2))
		#print(conv_fourth.shape, torch.min(conv_fourth).item(), torch.max(conv_fourth).item(), torch.mean(conv_fourth).item())
		conv_fifth = self.tanh(self.convolve(self.conv5, conv_fourth,  self.conv5_bias, stride=2))
		#print(conv_fifth.shape, torch.min(conv_fifth).item(), torch.max(conv_fifth).item(), torch.mean(conv_fifth).item())
		conv_sixth = self.tanh(self.convolve(self.conv6, conv_fifth,  self.conv6_bias, stride=2))
		#print(conv_sixth.shape, torch.min(conv_sixth).item(), torch.max(conv_sixth).item(), torch.mean(conv_sixth).item())
		
		return conv_sixth
		
		
		
	def convolve(self, conv: Tensor, input: Tensor, bias: Tensor,  stride=1):
		'''
		slides conv once over the input signal with stride = n and returns the result
		'''
		#print('convolution begins')
		width = len(conv)
		#print('input received', len(input))
		#print('width of filter', width)
		#print('stride', stride)
		#print('output dimension should be', ((len(input) - width) / stride) + 1)
		output = [torch.linalg.vecdot(conv, input[0 + stride * i : width + stride * i] ) + bias for i in range( int((len(input) - width) / stride + 1))]
		output = torch.stack(output).reshape(-1)
		#print(output.shape, output, 'min', torch.min(output).item(), 'max', torch.max(output).item())
		return output
		
	def rnn_forward(self, vecs: list[Tensor]):
		'''
		passes vecs through one pass of a bi-directional reccurrent network, and returns the output sequence as a list of probability distributions. 
		input shape: (n, 500)
		output shape: (n, 2)
		'''
		#print('RNN begins')
		prev = torch.ones((self.left_rnn_hidden.shape[0]),)
		hidden_list = []
		#print('left')
		for input in vecs:
			hidden = self.sigmoid(torch.matmul(self.left_rnn_in, input) + self.left_rnn_in_bias + torch.matmul(self.left_rnn_hidden, prev) + self.left_rnn_hidden_bias)
			#print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			hidden_list.append(hidden)
			prev = hidden
		reverse_hidden_list = []
		prev = torch.ones((self.left_rnn_hidden.shape[0]),)
		n = len(vecs) - 1
		#print('right')
		while n >= 0:
			input = vecs[n]
			hidden = self.sigmoid(torch.matmul(self.right_rnn_in, input) + self.right_rnn_in_bias + torch.matmul(self.right_rnn_hidden, prev) + self.right_rnn_hidden_bias)
			#print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			reverse_hidden_list.append(hidden)
			prev = hidden
			n -= 1
		hidden_list2 = []
		n = len(reverse_hidden_list) - 1
		while n >= 0:
			hidden_list2.append(reverse_hidden_list[n])
			n-=1
		full_context = []
		print('final layer begins')
		for hidden1, hidden2 in zip(hidden_list, hidden_list2):
			full = torch.cat((hidden1, hidden2))
			full_context.append(full)
		output_list = []
		for i, hidden in enumerate(full_context):
			first_layer = self.sigmoid(torch.matmul(self.output1, hidden) + self.output1_bias)
			#print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
			output = self.softmax(torch.matmul(self.output2, first_layer) + self.output2_bias)
			output_list.append(output)
		return torch.stack(output_list)
		
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
	
	epoch = 100
	while epoch > 0:
		optim.zero_grad()
		error = 0
		n = torch.randint(0, N1, (25,))
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
			
		
def read_in_data_test(batch: str):
	'''
	reads in input and gold data from the current directory, also reads in features
	'''
	print('reading in testing data')
	input_path, gold_path, feature_path = Path(f'data/vectors/{batch}/input'), Path(f'data/vectors/{batch}/gold.json'), Path(f'data/vectors/features/{batch}/input.json')
	input, gold = input_path, json.load(gold_path.open(mode='r'))

	return input, gold
	
		
def test_baseline(model, network: Network, input: list, gold: list):
	'''
	runs tests on the trained model and writes them to the current directory
	'''
	evaluate(model, network, input, 2, gold)
	evaluate(model, network, input, 3, gold)
	evaluate(model, network, input, 4, gold)
	
def evaluate(model, network: Network, input: str, syll: int, gold: list):
	'''
	passes, the input data through the network and uses the output to write an evaluation for the model performance
	'''
	print('evaluating model performance')
	output = []
	gold_list = []
	path = Path(f'{input}/meta.json')
	meta = json.load(path.open(mode='r'))
	files = meta[f'syllable_{syll}']['files']
	for file in files:
		print('received file', file)
		i = file[-1]
		n = file[0]
		print('reading in gold value', gold[n])
		gold_list.append(torch.tensor(gold[n][-1]))
		
		xpath = Path(f'{input}/{file[0]}_{file[1]}_{file[-1]}.json')
		waveform = torch.tensor(json.load(xpath.open(mode='r'))) 
		
		vecs = []
		model.eval()
		with torch.no_grad():
			vecs, _ = model.extract_features(waveform)
		x = []
		for vec in vecs[-1]:
			x.append(vec.reshape(-1))
		x = torch.stack(x)
		print(x.requires_grad)
		print(f'sequentially passing word {i} to the network', x.shape)
		y_hat = network.forward(x)
		print( y_hat)
		output.append(y_hat.detach())
		cycles = network.cycles
	r = recall(output, gold_list)
	p = precision(output, gold_list)
	f = fscore(r, p)
	write_results(cycles, syll, r, p, f, 'dev')
	write_feature_weights(output, gold_list, 'dev', input)
	
	
def write_results(cycles:int, syllables: int, r: float, p:float, fs: float, batch: str):
	'''
	writes the result to the directory 
	'''
	print('writing result table')
	path = Path(f'baseline/results/syllable_{syllables}/{batch}/model_performance.txt')
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
			
def write_feature_weights(output: list[Tensor], gold: list[Tensor], batch: str, input:str):
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
	path = Path(f'baseline/results/syllable_{syllable}/{batch}/feature_analysis.txt')
	print('writing to file', path)
	table_path = Path(f'data/data_{syllable}/{batch}/{batch}.csv')
	print('reading from table', table_path)
	table = pd.read_csv(table_path)
	table = table.set_index('Unnamed: 0')
	with path.open('w') as f:
		line = f'ipa\tsyllable\tprobability\tpredicted\tgold\tcorrect\n'
		f.write(line)
		for i, file in enumerate(files):
			index = file[-1]
			ipa = table['ipa'][index]
			for j in range(syllable):
				if j == 0:
					line = f'{ipa}\t{j + 1}\t{to_list(output[i][j])}\t{torch.argmax(output[i][j])}\t{gold[i][j]}\t{ gold[i][j] == torch.argmax(output[i][j])}\n'
					f.write(line)
				else:
					line = f'\t{j + 1}\t{to_list(output[i][j])}\t{torch.argmax(output[i][j])}\t{gold[i][j]}\t{ gold[i][j] == torch.argmax(output[i][j])}\n'
					f.write(line)
					
					
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
		network = Network()
		network.load_state_dict(model_parameters)
	bundle = torchaudio.pipelines.WAV2VEC2_BASE
	model = bundle.get_model()
	print('reading in training data')
	input, gold = read_in_data_train()
	print('input', input)
	print('gold shape', len(gold))
	train_baseline(model, network, input, gold)
	input, gold = read_in_data_test('dev')

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
