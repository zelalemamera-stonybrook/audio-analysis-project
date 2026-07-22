'''
This program implements the raw + praat model. This is equivalent to the raw model, with the exception that the output layer is provided weighted sum of praat features.
'''

import torch
import torchaudio
import json
from pathlib import Path
import torch.nn as nn
from torch import Tensor
import argparse
import pandas as pd
DEBUG = False

class Network(nn.Module):
	'''
	neural network implementation of the above
	'''
	def __init__(self):
		super().__init__()
		self.name = 'Raw+PraatModel'

		self.conv1 = nn.Conv1d(1, 3, (9,), stride = 5)
		self.conv1.weight.data = nn.init.uniform_(self.conv1.weight.data, -0.5 * 9, 0.5 * 9)

		self.conv2 = nn.Conv1d(3,2, (5,), stride=2)
		self.conv2.weight.data = nn.init.uniform_(self.conv2.weight.data, -0.5 * 5, 0.5 * 5)

		self.conv3 = nn.Conv1d(2, 1, (6,), stride=2)
		self.conv3.weight.data = nn.init.uniform_(self.conv3.weight.data, -0.5 * 6, 0.5 * 6)

		self.conv4 = nn.Conv1d(1,1,(4,), stride = 2)
		self.conv4.weight.data = nn.init.uniform_(self.conv4.weight.data, -0.5 * 4, 0.5 * 4)

		self.conv5 = nn.Conv1d(1,1, (4,), stride = 2)
		self.conv5.weight.data = nn.init.uniform_(self.conv5.weight.data, -0.5 * 4, 0.5 * 4)

		self.conv6 = nn.Conv1d(1,1, (4,), stride = 2)
		self.conv6.weight.data = nn.init.uniform_(self.conv6.weight.data, -0.5 * 4, 0.5 * 4)

		self.attnlayer1 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((500, 1000 + 225)),  - 0.5, 0.5))
		self.attnlayer1_bias = nn.parameter.Parameter(torch.rand((500)) - 0.5)
		self.attnlayer2 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((100, 500) ), - 0.5, 0.5))
		self.attnlayer2_bias = nn.parameter.Parameter(torch.rand((100) ) - 0.5)
		self.attnlayer3 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1,100)), - 0.5 * 1.5, 0.5 * 1.5))
		self.attnlayer3_bias = nn.parameter.Parameter(torch.rand((1,)) - 0.5)

		self.left_rnn_in = nn.parameter.Parameter(torch.rand((500,314)) - 0.5)
		self.left_rnn_in_bias = nn.parameter.Parameter(torch.rand((500,))- 0.5)
		self.left_rnn_hidden = nn.parameter.Parameter(torch.rand((500,500))-0.5)
		self.left_rnn_hidden_bias = nn.parameter.Parameter(torch.rand((500,))-0.5)

		self.right_rnn_in = nn.parameter.Parameter(torch.rand((500,314))-0.5)
		self.right_rnn_in_bias = nn.parameter.Parameter(torch.rand((500,))-0.5)
		self.right_rnn_hidden = nn.parameter.Parameter(torch.rand((500,500))-0.5)
		self.right_rnn_hidden_bias = nn.parameter.Parameter(torch.rand((500,))-0.5)

		self.output1 = nn.parameter.Parameter(torch.rand((500, 1000 + 225))-0.5)
		self.output1_bias = nn.parameter.Parameter(torch.rand((500,))-0.5)

		self.output2 = nn.parameter.Parameter(torch.rand((200, 500))-0.5)
		self.output2_bias = nn.parameter.Parameter(torch.rand((200,))-0.5)

		self.output3 = nn.parameter.Parameter(torch.rand((2, 200))-0.5)
		self.output3_bias = nn.parameter.Parameter(torch.rand((2,))-0.5)

		self.attention_weights = []
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim = 0)

	def forward(self, input: list[Tensor], features=None):
		'''
		the input is a list of vectors, these are first passed each to the MLP which converts them to 500, then the list is processed by a bidirectional rnn that outputs the probability distribution for each syllable
		'''
		output_list = []
		for vec in input:
			output_list.append(self.convolution_forward(vec))
		output = self.rnn_forward(output_list, features)
		return output



	def attend(self, hidden: Tensor, feature_vecs: Tensor):
		'''
		computes the attention score of each element in the list with respect to the other elements, then returns the weighted sum of the whole
		input shape: (400) + (30) * 4
		output shape: (400)
		'''
		if DEBUG:
			print('starting attention network')
		weight_list = []
		if DEBUG:
			print('compatibility is computed over', feature_vecs.shape)
		for attention_target in feature_vecs:
			weight_list.append(self.attention_forward(hidden, attention_target))
		weight_tensor = torch.stack(weight_list).reshape(-1)
		attention_vector = self.softmax(weight_tensor)
		print(attention_vector)
		weighted = self.sigmoid(torch.matmul(attention_vector, feature_vecs))
		if DEBUG:
			print(weighted.shape, torch.min(weighted).item(), torch.max(weighted).item(), torch.mean(weighted).item())
		output = torch.cat((hidden, weighted))
		return attention_vector, output


	def attention_forward(self, attention_source: Tensor, attention_target: Tensor):
		'''
		computes the compatibility score of the source to the target
		input shape: (400 + 30)
		output shape: (1)
		'''
		if DEBUG:
			print('attention forward begins')
		input = torch.cat((attention_source, attention_target))
		if DEBUG:
			print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
		first_layer = self.sigmoid(torch.matmul( self.attnlayer1, input ) + self.attnlayer1_bias)
		if DEBUG:
			print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
		second_layer = self.sigmoid(torch.matmul(self.attnlayer2, first_layer ) + self.attnlayer2_bias)
		if DEBUG:
			print(second_layer.shape, torch.min(second_layer).item(), torch.max(second_layer).item(), torch.mean(second_layer).item())
		third_layer = torch.matmul(self.attnlayer3, second_layer ) + self.attnlayer3_bias
		if DEBUG:
			print(third_layer)
		return third_layer


	def filter(self, features: Tensor, i: int):
		'''
		returns the ith element of each tensor in features
		'''
		filtered_list = []
		for feature in features:
			filtered_list.append(feature[i])
		return torch.stack(tuple(filtered_list))


	def convolution_forward(self, vec: Tensor):
		'''
		passes vec through two convolution layers as specified and returns the result
		'''
		if DEBUG:
			print('convolution layer begins')
		conv_first = self.tanh(self.conv1(vec.reshape((1, vec.shape[0]))))
		if DEBUG:
			print(conv_first.shape, torch.min(conv_first).item(), torch.max(conv_first).item(), torch.mean(conv_first).item())

		conv_second = self.tanh(self.conv2(conv_first))
		if DEBUG:
			print(conv_second.shape, torch.min(conv_second).item(), torch.max(conv_second).item(), torch.mean(conv_second).item())

		conv_third = self.tanh(self.conv3(conv_second))
		if DEBUG:
			print(conv_third.shape, torch.min(conv_third).item(), torch.max(conv_third).item(), torch.mean(conv_third).item())

		conv_fourth = self.tanh(self.conv4(conv_third))
		if DEBUG:
			print(conv_fourth.shape, torch.min(conv_fourth).item(), torch.max(conv_fourth).item(), torch.mean(conv_fourth).item())

		conv_fifth = self.tanh(self.conv5(conv_fourth))
		if DEBUG:
			print(conv_fifth.shape, torch.min(conv_fifth).item(), torch.max(conv_fifth).item(), torch.mean(conv_fifth).item())

		conv_sixth = self.sigmoid(self.conv6(conv_fifth))
		if DEBUG:
			print(conv_sixth.shape, torch.min(conv_sixth).item(), torch.max(conv_sixth).item(), torch.mean(conv_sixth).item())

		return conv_sixth.reshape(-1)


	def convolve(self, conv: Tensor, input: Tensor, bias: Tensor,  stride=1):
		'''
		slides conv once over the input signal with stride = n and returns the result
		'''
		if DEBUG:
			print('convolution begins')
		width = len(conv)
		if DEBUG:
			print('input received', len(input))
			print('width of filter', width)
			print('stride', stride)
			print('output dimension should be', ((len(input) - width) / stride) + 1)
		output = [torch.linalg.vecdot(conv, input[0 + stride * i : width + stride * i] ) + bias for i in range( int((len(input) - width) / stride + 1))]
		output = torch.stack(output).reshape(-1)
		if DEBUG:
			print(output.shape, output, 'min', torch.min(output).item(), 'max', torch.max(output).item())
		return output

	def rnn_forward(self, vecs: list[Tensor], features=None):
		'''
		passes vecs through one pass of a bi-directional reccurrent network, and returns the output sequence as a list of probability distributions.
		input shape: (n, 500)
		output shape: (n, 2)
		'''
		if DEBUG:
			print('RNN begins')
		prev = torch.ones((self.left_rnn_hidden.shape[0]),)
		hidden_list = []
		if DEBUG:
			print('left')
		for input in vecs:
			hidden = self.sigmoid(torch.matmul(self.left_rnn_in, input) + self.left_rnn_in_bias + torch.matmul(self.left_rnn_hidden, prev) + self.left_rnn_hidden_bias)
			if DEBUG:
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			hidden_list.append(hidden)
			prev = hidden
		reverse_hidden_list = []
		prev = torch.ones((self.left_rnn_hidden.shape[0]),)
		n = len(vecs) - 1
		if DEBUG:
			print('right')
		while n >= 0:
			input = vecs[n]
			hidden = self.sigmoid(torch.matmul(self.right_rnn_in, input) + self.right_rnn_in_bias + torch.matmul(self.right_rnn_hidden, prev) + self.right_rnn_hidden_bias)
			if DEBUG:
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			reverse_hidden_list.append(hidden)
			prev = hidden
			n -= 1
		hidden_list2 = []
		n = len(reverse_hidden_list) - 1
		while n >= 0:
			hidden_list2.append(reverse_hidden_list[n])
			n-=1
		full_context = []
		if DEBUG:
			print('final layer begins')
		for hidden1, hidden2 in zip(hidden_list, hidden_list2):
			full = torch.cat((hidden1, hidden2))
			full_context.append(full)
		output_list = []
		if features:
			attention_list = []
		for i, hidden in enumerate(full_context):
			if features:
				feature_vecs = self.filter(features, i)
				attention_vector, hidden = self.attend(hidden, feature_vecs)
				attention_list.append(attention_vector.tolist())
			first_layer = self.sigmoid(torch.matmul(self.output1, hidden) + self.output1_bias)
			if DEBUG:
				print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
			output2 = self.sigmoid(torch.matmul(self.output2, first_layer) + self.output2_bias)
			output = self.softmax(torch.matmul(self.output3, output2) + self.output3_bias)
			output_list.append(output)
		if features:
			self.attention_weights.append(attention_list)
		return torch.stack(output_list)
