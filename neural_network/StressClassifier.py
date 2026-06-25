'''
The following code specifies a neural network that takes as its input a word (treated as a sequence of syllables) and outputs a sequence of probability distributions (one for each syllable). 
We try to use some sort of attention mechanism to infer the importance of the linguistic features used. 
'''

import torch
import torchaudio
import torch.nn as nn
from torch import Tensor

class Network(nn.Module):
	'''
	neural network implementation for the above
	'''
	def __init__(self):
		super().__init__()
		print('initializing parameters')
		self.cycles = nn.parameter.Parameter(torch.tensor(float(0)), requires_grad = False) 
		self.conv1 = nn.parameter.Parameter((torch.rand((10,)) - .5) * 10,  requires_grad = True) # stride = 5
		self.conv1_bias = nn.parameter.Parameter(torch.rand((1,)), requires_grad = True)
		self.conv2 = nn.parameter.Parameter((torch.rand((5,)) - 0.5) * 5, requires_grad = True) # stride = 3
		self.conv2_bias = nn.parameter.Parameter(torch.rand((1,)), requires_grad = True)
		self.conv3 = nn.parameter.Parameter((torch.rand((21,)) - 0.5) * 21, requires_grad = True) # stride = 2
		self.conv3_bias = nn.parameter.Parameter(torch.rand((1,)), requires_grad = True)
		
		self.feature_encoder_in = nn.parameter.Parameter(torch.rand((990, 90)) - 0.5, requires_grad = True)
		self.feature_encoder_in_bias = nn.parameter.Parameter(torch.rand((990)) -0.5, requires_grad = True)
		self.feature_encoder_hidden = nn.parameter.Parameter(torch.rand((990, 990) )- 0.5, requires_grad = True)
		self.feature_encoder_hidden_bias = nn.parameter.Parameter(torch.rand((990) )- 0.5, requires_grad = True)
		
		self.combine_layer1 = nn.parameter.Parameter(torch.rand((990, 990 * 2)) - 0.5, requires_grad = True)
		self.combine_layer1_bias = nn.parameter.Parameter(torch.rand((990)) - 0.5, requires_grad = True)
		self.combine_layer2 = nn.parameter.Parameter(torch.rand((990, 990)) - 0.5, requires_grad = True)
		self.combine_layer2_bias  = nn.parameter.Parameter(torch.rand((990)) - 0.5, requires_grad = True)
	
		self.attnlayer1 = nn.parameter.Parameter(torch.rand((1000, 1000 * 2)) - 0.5, requires_grad = True)
		self.attnlayer1_bias = nn.parameter.Parameter(torch.rand((1000) )- 0.5, requires_grad = True)
		self.attnlayer2 = nn.parameter.Parameter(torch.rand((100,1000) )- 0.5, requires_grad = True)
		self.attnlayer2_bias = nn.parameter.Parameter(torch.rand((100) )- 0.5, requires_grad = True)
		self.attnlayer3 = nn.parameter.Parameter(torch.rand((1,100)) - 0.5, requires_grad = True)
		self.attnlayer3_bias = nn.parameter.Parameter(torch.rand((1,)) - 0.5, requires_grad = True)
		self.feature_weights = []
			
		self.attention_combine = nn.parameter.Parameter(torch.rand((1, 4)) - 0.5, requires_grad = True)
		self.attention_combine_bias = nn.parameter.Parameter(torch.rand((1,)) - 0.5, requires_grad = True)
		
		self.encode_in = nn.parameter.Parameter(torch.rand((500, 90) )- 0.5, requires_grad = True)
		self.encode_in_bias = nn.parameter.Parameter(torch.rand((500) )- 0.5, requires_grad = True)
		self.encode_hidden = nn.parameter.Parameter(torch.rand((500, 500)) - 0.5, requires_grad = True)
		self.encode_hidden_bias = nn.parameter.Parameter(torch.rand((500)) - 0.5, requires_grad = True)
	
		self.recurrent_left_in = nn.parameter.Parameter(torch.rand((500, 500) )-0.5, requires_grad = True)
		self.recurrent_left_in_bias = nn.parameter.Parameter(torch.rand((500) )-0.5, requires_grad = True)
		self.recurrent_left_hidden = nn.parameter.Parameter(torch.rand((500,500)) -0.5, requires_grad = True)
		self.recurrent_left_hidden_bias = nn.parameter.Parameter(torch.rand((500)) -0.5, requires_grad = True)
		
		self.recurrent_right_in = nn.parameter.Parameter(torch.rand((500,500)) -0.5, requires_grad = True)
		self.recurrent_right_in_bias = nn.parameter.Parameter(torch.rand((500) )-0.5, requires_grad = True)
		self.recurrent_right_hidden = nn.parameter.Parameter(torch.rand((500,500)) -0.5, requires_grad = True)
		self.recurrent_right_hidden_bias = nn.parameter.Parameter(torch.rand((500)) -0.5, requires_grad = True)
		
		self.recurrent_out1 = nn.parameter.Parameter(torch.rand((100, 1000)) -0.5, requires_grad = True)
		self.recurrent_out1_bias = nn.parameter.Parameter(torch.rand((100)) -0.5, requires_grad = True)
		self.recurrent_out2 = nn.parameter.Parameter(torch.rand((2, 100) )-0.5, requires_grad = True)
		self.recurrent_out2_bias = nn.parameter.Parameter(torch.rand((2)) -0.5, requires_grad = True)

		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=-1)
					
	def forward(self, word: Tensor, features: Tensor, debug=False):
		'''
		passes the word once through the network, and returns the output. 
		input shape: (n, 30,000)
		output shape: (n, 2)
		where n is the number of syllables >= 2
		features is a list of feature embeddings of this word
		'''
		if debug:
			return torch.zeros((word.shape[0], 2))
		sound_vec_embedding = []
		#print('embedding syllables')
		for syll in word:
			#print('syll received', syll.shape)
			sound_vec_embedding.append(self.embed(syll))
			#print('syllable embedded')
			#analyze_graph(sound_vec_embedding[-1])
		word_encoding = []
		#print('encoding vectors')
		for syll in sound_vec_embedding:
			word_encoding.append(self.encode(syll))
			#print('vector encoded')
		#print('starting bi-directional recurrent network')
		output = self.rnn_forward(torch.stack(word_encoding), features)
		feature_injected_vecs = []
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
			
	def encode(self, syll: Tensor):
		'''
		runs the encoder over this tensor
		input shape (990,)
		output shape (500)
		'''
		#print('encoder begins')
		prev = torch.ones((self.encode_hidden.shape[0],))
		n = 0
		length =  int(len(syll) / self.encode_in.shape[1])
		#print('passing through vector')
		for i in range(length):
			input = syll[n: n + self.encode_in.shape[1]]
			#print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			hidden = self.sigmoid(torch.matmul(self.encode_in, input) + self.encode_in_bias + torch.matmul(self.encode_hidden, prev) + self.encode_hidden_bias)
			#print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			prev = hidden
			n += self.encode_in.shape[1]
		return prev
			
		
		
	def embed(self, input: Tensor):
		'''
		embeds the input tensor into a compact representation, we first add a channel since the input shape is channelless
		input shape: (30,000)
		output shape: (990)
		'''
		conv_first = self.tanh(self.convolve(self.conv1, input, self.conv1_bias, stride=5))
		#print(conv_first.shape, torch.min(conv_first).item(), torch.max(conv_first).item(), torch.mean(conv_first).item())
		conv_second = self.tanh(self.convolve(self.conv2, conv_first,  self.conv2_bias, stride=3))
		#print(conv_second.shape, torch.min(conv_second).item(), torch.max(conv_second).item(), torch.mean(conv_second).item())
		conv_third = self.tanh(self.convolve(self.conv3, conv_second, self.conv3_bias, stride=2))
		#print(conv_third.shape, torch.min(conv_third).item(), torch.max(conv_third).item(), torch.mean(conv_third).item())
		
		return conv_third
		
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
		
	def inject_features(self, embedding: Tensor, features: Tensor):
		'''
		injects our linguistic features into the embedding, then computes self attention weights with respect to the sequence and returns the weighted sum. 
		input shape: (990)
		output shape: (990)
		'''
		#print('attention network begins')
		injected_list = []
		for feature in features:
			encoded_feature = self.encode_features(feature)
			combined = self.combine(embedding, encoded_feature)
			injected_list.append(combined)
		attended = self.attend(torch.stack(injected_list))
		return attended
		
	
	def combine(self, embedding: Tensor, encoded_feature: Tensor):
		'''
		passes the two tensors through the combiner network and returns the result
		input shape (990 * 2)
		output shape (990)
		'''
		#print('combiner network begins')
		input = torch.cat((embedding, encoded_feature))
		#print(input.shape)
		first_layer = self.sigmoid(torch.matmul(self.combine_layer1, input) + self.combine_layer1_bias)
		#print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
		second_layer = self.sigmoid(torch.matmul(self.combine_layer2, first_layer) + self.combine_layer2_bias)
		#print(second_layer.shape, torch.min(second_layer).item(), torch.max(second_layer).item(), torch.mean(second_layer).item())
		#print('combined')
		#print(second_layer.shape)
		return second_layer
		

	def encode_features(self, feature: Tensor):
		'''
		passes the a feature encoder over the input tensor and returns the result
		input shape (30000)
		output shape (990)
		'''
		#print('feature encoder begins')
		prev = torch.ones((self.feature_encoder_hidden.shape[0],))
		n = 0
		length =  int(len(feature) / self.feature_encoder_in.shape[1])
		#print('encoding feature')
		for i in range(length):
			input = feature[n: n + self.feature_encoder_in.shape[1]]
			hidden = self.sigmoid(torch.matmul(self.feature_encoder_in, input) + self.feature_encoder_in_bias + torch.matmul(self.feature_encoder_hidden, prev) + self.feature_encoder_hidden_bias)
			#print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			prev = hidden
			n += self.feature_encoder_in.shape[1]
		return prev
		
	def attend(self, hidden: Tensor, feature_vecs: Tensor):
		'''
		computes the attention score of each element in the list with respect to the other elements, then returns the weighted sum of the whole
		input shape: (1000) + (1000) * 4
		output shape: (1000)
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
		output = self.sigmoid(hidden + weighted)
		return attention_vector, output
		
	def attn_combine(self, weights: Tensor):
		'''
		each vector generates n attention scores based on its compatibility with the other vectors. However, only one score needs to be used to weigh a single vector. this network
		produces such a number.
		input shape (4,)
		output shape (1,)
		'''
		output = self.sigmoid(torch.matmul(self.attention_combine, weights) + self.attention_combine_bias)
		return output
		
	
	def attention_forward(self, attention_source: Tensor, attention_target: Tensor):
		'''
		computes the compatibility score of the source to the target
		input shape: (1000 * 2)
		output shape: (1)
		'''
		#print('attention forward begins')
		input = torch.cat((attention_source, attention_target))
		#print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
		first_layer = self.sigmoid(torch.matmul( self.attnlayer1, input ) + self.attnlayer1_bias)
		#print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
		second_layer = self.sigmoid(torch.matmul(self.attnlayer2, first_layer ) + self.attnlayer2_bias)
		#print(second_layer.shape, torch.min(second_layer).item(), torch.max(second_layer).item(), torch.mean(second_layer).item())
		third_layer = self.sigmoid(torch.matmul(self.attnlayer3, second_layer ) + self.attnlayer3_bias)
		#print(third_layer)
		return third_layer
		
	def rnn_forward(self, injected_list: Tensor, features: Tensor):
		'''
		passes the list of feature injected and attention weighted tensors through one pass of a bi-directional reccurrent network, and returns the output sequence as a list of probability distributions over the two classes. 
		input shape: (n, 500)
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
			#print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			#print('prediction layer')
			first_layer = self.sigmoid(torch.matmul(self.recurrent_out1, input) + self.recurrent_out1_bias)
			#print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
			second_layer = self.softmax(torch.matmul(self.recurrent_out2, first_layer) + self.recurrent_out2_bias)
			output_list.append(second_layer)
			#print(second_layer.shape, second_layer)
		self.feature_weights.append(attention_list)
		return torch.stack(output_list)
	
			
			
			
		
				
		

	
	
	