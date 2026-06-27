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
		
		self.conv1 = nn.Conv1d(1, 3, (9,), stride = 5)
		self.conv1.weight.data = nn.init.uniform_(self.conv1.weight.data, -9 * 0.5, 9 * 0.5)
		
		self.conv2 = nn.Conv1d(3,2, (5,), stride=2) 
		self.conv2.weight.data = nn.init.uniform_(self.conv2.weight.data, -5 * 0.5, 5 * 0.5)
		
		self.conv3 = nn.Conv1d(2, 1, (6,), stride=2)
		self.conv3.weight.data = nn.init.uniform_(self.conv3.weight.data, -6 * 0.5, 6 * 0.5)
		
		self.conv4 = nn.Conv1d(1,1,(4,), stride = 2)
		self.conv4.weight.data = nn.init.uniform_(self.conv4.weight.data, -4 * 0.5, 4 * 0.5)
		
		self.conv5 = nn.Conv1d(1,1, (4,), stride = 2)
		self.conv5.weight.data = nn.init.uniform_(self.conv5.weight.data, -4 * 0.5, 4 * 0.5) 
		
		self.conv6 = nn.Conv1d(1,1, (4,), stride = 2)
		self.conv6.weight.data = nn.init.uniform_(self.conv6.weight.data, -4 * 0.5, 4 * 0.5)
		
		
		self.left_rnn_in = nn.parameter.Parameter(torch.rand((500,444)) - 0.5)
		self.left_rnn_in_bias = nn.parameter.Parameter(torch.rand((500,)))
		self.left_rnn_hidden = nn.parameter.Parameter(torch.rand((500,500)) - 0.5)
		self.left_rnn_hidden_bias = nn.parameter.Parameter(torch.rand((500,)))
	
		self.right_rnn_in = nn.parameter.Parameter(torch.rand((500,444)) - 0.5)
		self.right_rnn_in_bias = nn.parameter.Parameter(torch.rand((500,)))
		self.right_rnn_hidden = nn.parameter.Parameter(torch.rand((500,500)) - 0.5)
		self.right_rnn_hidden_bias = nn.parameter.Parameter(torch.rand((500,)))
	
		self.output1 = nn.parameter.Parameter(torch.rand((500, 1000)) - 0.5)
		self.output1_bias = nn.parameter.Parameter(torch.rand((500,)))
		
		self.output2 = nn.parameter.Parameter(torch.rand((2, 500)) - 0.5)
		self.output2_bias = nn.parameter.Parameter(torch.rand((2,)))
		
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
		conv_first = self.tanh(self.conv1(vec.reshape((1, vec.shape[0]))))
		#print(conv_first.shape, torch.min(conv_first).item(), torch.max(conv_first).item(), torch.mean(conv_first).item())
		
		conv_second = self.tanh(self.conv2(conv_first))
		#print(conv_second.shape, torch.min(conv_second).item(), torch.max(conv_second).item(), torch.mean(conv_second).item())
		
		conv_third = self.tanh(self.conv3(conv_second)	)	
		#print(conv_third.shape, torch.min(conv_third).item(), torch.max(conv_third).item(), torch.mean(conv_third).item())
		
		conv_fourth = self.tanh(self.conv4(conv_third))
		#print(conv_fourth.shape, torch.min(conv_fourth).item(), torch.max(conv_fourth).item(), torch.mean(conv_fourth).item())
		
		conv_fifth = self.tanh(self.conv5(conv_fourth))
		#print(conv_fifth.shape, torch.min(conv_fifth).item(), torch.max(conv_fifth).item(), torch.mean(conv_fifth).item())
		
		conv_sixth = self.sigmoid(self.conv6(conv_fifth))
		#print(conv_sixth.shape, torch.min(conv_sixth).item(), torch.max(conv_sixth).item(), torch.mean(conv_sixth).item())
		
		return conv_sixth.reshape(-1)
		
		
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
		#print('final layer begins')
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
			
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
