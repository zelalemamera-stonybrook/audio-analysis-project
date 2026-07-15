'''
The following code specifies a neural network that takes as its input a word (treated as a sequence of syllables) and outputs a sequence of probability distributions (one for each syllable).
We try to use some sort of attention mechanism to infer the importance of the linguistic features used.
'''

import torch
import torchaudio
import torch.nn as nn
from torch import Tensor
DEBUG = True

class Network(nn.Module):
	'''
	neural network implementation for the above
	'''
	def __init__(self):
		super().__init__()
		print('initializing parameters')
		self.epochs = nn.parameter.Parameter(torch.tensor(float(0)), requires_grad = False)

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


		self.attnlayer1 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((500, 2000 + 165)),  - 0.5, 0.5))
		self.attnlayer1_bias = nn.parameter.Parameter(torch.rand((500)) - 0.5)
		self.attnlayer2 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((100, 500) ), - 0.5, 0.5))
		self.attnlayer2_bias = nn.parameter.Parameter(torch.rand((100) ) - 0.5)
		self.attnlayer3 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1,100)), - 0.5 * 1.5, 0.5 * 1.5))
		self.attnlayer3_bias = nn.parameter.Parameter(torch.rand((1,)) - 0.5)
		self.feature_weights = []

		self.recurrent_left_in = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1000, 165 * 13) ),  - 0.5, 0.5))
		self.recurrent_left_in_bias = nn.parameter.Parameter(torch.rand((1000) ))
		self.recurrent_left_hidden = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1000,1000)), -0.5, 0.5))
		self.recurrent_left_hidden_bias = nn.parameter.Parameter(torch.rand((1000)))

		self.recurrent_right_in = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1000,165 * 13)), - 0.5, 0.5))
		self.recurrent_right_in_bias = nn.parameter.Parameter(torch.rand((1000) ))
		self.recurrent_right_hidden = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1000,1000)), -0.5, 0.5))
		self.recurrent_right_hidden_bias = nn.parameter.Parameter(torch.rand((1000)))

		self.recurrent_out1 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1000, 2000 + 165)), - 0.5, 0.5))
		self.recurrent_out1_bias = nn.parameter.Parameter(torch.rand((1000)))
		self.recurrent_out2 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((500, 1000) ), - 0.5, 0.5))
		self.recurrent_out2_bias = nn.parameter.Parameter(torch.rand((500,)))
		self.recurrent_out3 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((100, 500) ), - 0.5, 0.5))
		self.recurrent_out3_bias = nn.parameter.Parameter(torch.rand((100,)))
		self.recurrent_out4 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((2, 100) ), - 0.5, 0.5))
		self.recurrent_out4_bias = nn.parameter.Parameter(torch.rand((2,)))

		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, word: list, features: list):
		'''
		passes the word once through the network, and returns the output
		word shape: (n, 30000)
		features shape: (f, n, 250)
		output shape: (n, 2)
		where n is the number of syllables >= 2
		'''
		'''
		sound_vec_embedding = []
		#print('embedding syllables')
		for syll in word:
			#print('syll received', syll.shape)
			sound_vec_embedding.append(self.convolution_forward(syll))
			#print('syllable embedded')
			#analyze_graph(sound_vec_embedding[-1])
		#print('starting bi-directional recurrent network')
		'''
		output = self.rnn_forward(word, features)
		print('finished forward pass')
		return output


	def filter(self, features: Tensor, i: int):
		'''
		returns the ith element of each tensor in features
		'''
		filtered_list = []
		for word in features:
			filtered_list.append(word[i])
		return torch.stack(tuple(filtered_list))
			
		
		
	def convolution_forward(self, vec: Tensor):
		'''
		embeds the input tensor into a compact representation, we first add a channel since the input shape is channelless
		input shape: (30,000)
		output shape: (990)
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
		

		
	def attend(self, hidden: Tensor, feature_vecs: Tensor):
		'''
		computes the attention score of each element in the list with respect to the other elements, then returns the weighted sum of the whole
		input shape: (400) + (30) * 4
		output shape: (400)
		'''
		#print('starting attention network')
		weight_list = []
		#print('compatibility is computed over', feature_vecs.shape)
		for attention_target in feature_vecs:
			weight_list.append(self.attention_forward(hidden, attention_target))
		weight_tensor = torch.stack(weight_list).reshape(-1)
		attention_vector = self.softmax(weight_tensor)
		print(attention_vector)
		weighted = torch.matmul(attention_vector, feature_vecs)
		#print(weighted.shape, torch.min(weighted).item(), torch.max(weighted).item(), torch.mean(weighted).item())
		output = torch.cat((hidden, weighted))
		return attention_vector, output
		
		
	
	def attention_forward(self, attention_source: Tensor, attention_target: Tensor):
		'''
		computes the compatibility score of the source to the target
		input shape: (400 + 30)
		output shape: (1)
		'''
		#print('attention forward begins')
		input = torch.cat((attention_source, attention_target))
		#print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
		first_layer = self.sigmoid(torch.matmul( self.attnlayer1, input ) + self.attnlayer1_bias)
		#print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
		second_layer = self.sigmoid(torch.matmul(self.attnlayer2, first_layer ) + self.attnlayer2_bias)
		#print(second_layer.shape, torch.min(second_layer).item(), torch.max(second_layer).item(), torch.mean(second_layer).item())
		third_layer = torch.matmul(self.attnlayer3, second_layer ) + self.attnlayer3_bias
		#print(third_layer)
		return third_layer
		
	def rnn_forward(self, injected_list: Tensor, features: Tensor):
		'''
		passes the list of feature injected and attention weighted tensors through one pass of a bi-directional reccurrent network, and returns the output sequence as a list of probability distributions over the two classes. 
		input shape: (n, 30)
		output shape: (n, 2)
		'''
		#print('final layer begins')
		prev = torch.ones((self.recurrent_left_hidden.shape[0]),)
		hidden_list = []
		#print('left rnn')
		for input in injected_list:
			#print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			hidden = self.sigmoid(torch.matmul(self.recurrent_left_in, input) + self.recurrent_left_in_bias + torch.matmul(self.recurrent_left_hidden, prev) + self.recurrent_left_hidden_bias)
			#print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			hidden_list.append(hidden)
			prev = hidden
		reverse_hidden_list = []
		prev = torch.ones((self.recurrent_left_hidden.shape[0]),)
		n = len(injected_list) - 1
		#print('right rnn')
		while n >= 0:
			input = injected_list[n]
			#print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			hidden = self.sigmoid(torch.matmul(self.recurrent_right_in, input) + self.recurrent_right_in_bias + torch.matmul(self.recurrent_right_hidden, prev) + self.recurrent_right_hidden_bias)
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
		for hidden1, hidden2 in zip(hidden_list, hidden_list2):
			full = torch.cat((hidden1, hidden2))
			#print(full.shape, torch.min(full).item(), torch.max(full).item(), torch.mean(full).item())
			full_context.append(full)
		output_list = []
		#print('output layer')
		attention_list = []
		for i, hidden in enumerate(full_context):
			#print('received hidden vector')
			#print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			feature_vecs = self.filter(features, i)
			attention_vector, input = self.attend(hidden, feature_vecs)
			attention_list.append(attention_vector.tolist())
			#print('weighted hidden vector')
			#print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			#print('prediction layer')
			first_layer = self.sigmoid(torch.matmul(self.recurrent_out1, input) + self.recurrent_out1_bias)
			#print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
			second_layer = self.sigmoid(torch.matmul(self.recurrent_out2, first_layer) + self.recurrent_out2_bias)
			#print(second_layer.shape, torch.min(second_layer).item(), torch.max(second_layer).item(), torch.mean(second_layer).item())
			third_layer = self.sigmoid(torch.matmul(self.recurrent_out3, second_layer) + self.recurrent_out3_bias)
			#print(third_layer.shape, torch.min(third_layer).item(), torch.max(third_layer).item(), torch.mean(third_layer).item())
			fourth_layer = self.softmax(torch.matmul(self.recurrent_out4, third_layer) + self.recurrent_out4_bias)
			output_list.append(fourth_layer)
		self.feature_weights.append(attention_list)
		return torch.stack(output_list)
	
			
			
			
		
				
		

	
	
	
