'''
The following code specifies a deep learning architecture that classifies stress position of an input audio file. Due to the high dimensionality of the audio data, and the variability of the length of the input vectors, we pass the audio through 
a convolutional filter to extract higher level representations, followed by encoding. The inputs of each sound file are thus a sequence of vector embeddings representing the sequence of syllables in the word. This sequence is
presented to a bi directional RNN which uses the full contedxtual information of the sequence to classify each of the syallbles as stressed or unstressed. In order to introduce some interpretibility to the model, for each syllable embedding we inject linguisticly relevant
spectral features. Specifically, for any n features that we can generate from the syllable, we make n copies of the embedding and inject each with a separate feature, then take a linear combination of those. this provides the model with the same information altered perhaps by the presence of a specific 
feature. In order to measure the impact that any such feature might have, we make the weights learnable functions of the input. In particular, we combine the input sequence with a k fully connected layers that produce the relevant weight
for each vector in the sequence. These weights are therefore learned as part of the network and can later be studied to potentially infer the importance of a feature.
'''

import torch
import torchaudio
import torch.nn as nn
from torch import Tensor

class Network(nn.Module):
	'''
	neural network implementation for the above
	'''
	def __init__(self, parameters=None):
		super().__init__()
		
		if parameters:
			print('loading in parameters')
			self.load_state_dict(parameters)
		else:
			print('initializing parameters')
			self.cycles = nn.parameter.Parameter(torch.tensor(float(0)), requires_grad = False)
			self.conv1 = nn.Conv1d(1,1,10,5)
			self.conv2 = nn.Conv1d(1,1, 5, 3) 
			self.conv3 = nn.Conv1d(1,1, 21, 2)
	
			self.attnlayer1 = nn.parameter.Parameter(torch.rand((2,1980), requires_grad = True))
			self.attnlayer1_bias = nn.parameter.Parameter(torch.rand((2), requires_grad = True))
			self.attnlayer2 = nn.parameter.Parameter(torch.rand((1,2), requires_grad = True))
			self.attnlayer2_bias = nn.parameter.Parameter(torch.rand((1), requires_grad = True))
			self.feature_weights = []
			
			self.encode_in = nn.parameter.Parameter(torch.rand((500, 90), requires_grad = True))
			self.encode_in_bias = nn.parameter.Parameter(torch.rand((500), requires_grad = True))
			self.encode_hidden = nn.parameter.Parameter(torch.rand((500, 500), requires_grad = True))
			self.encode_hidden_bias = nn.parameter.Parameter(torch.rand((500), requires_grad = True))
	
			self.recurrent_left_in = nn.parameter.Parameter(torch.rand((1000, 500), requires_grad = True))
			self.recurrent_left_in_bias = nn.parameter.Parameter(torch.rand((1000), requires_grad = True))
			self.recurrent_left_hidden = nn.parameter.Parameter(torch.rand((1000,1000), requires_grad = True))
			self.recurrent_left_hidden_bias = nn.parameter.Parameter(torch.rand((1000), requires_grad = True))
		
			self.recurrent_right_in = nn.parameter.Parameter(torch.rand((1000,500), requires_grad = True))
			self.recurrent_right_in_bias = nn.parameter.Parameter(torch.rand((1000), requires_grad = True))
			self.recurrent_right_hidden = nn.parameter.Parameter(torch.rand((1000,1000), requires_grad = True))
			self.recurrent_right_hidden_bias = nn.parameter.Parameter(torch.rand((1000), requires_grad = True))
		
			self.recurrent_out1 = nn.parameter.Parameter(torch.rand((100, 2000), requires_grad = True))
			self.recurrent_out1_bias = nn.parameter.Parameter(torch.rand((100), requires_grad = True))
			self.recurrent_out2 = nn.parameter.Parameter(torch.rand((2, 100), requires_grad = True))
			self.recurrent_out2_bias = nn.parameter.Parameter(torch.rand((2), requires_grad = True))
		
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
		print('embedding syllables')
		for syll in word:
			print('syll dimension', syll.shape)
			sound_vec_embedding.append(self.embed(syll))
			print('syllable embedded', sound_vec_embedding[-1].shape)
		feature_injected_vecs = []
		print('injecting features')
		for i, syll in enumerate(sound_vec_embedding):
			syll_features = self.filter(features, i)
			print('features to be injected', syll_features.shape)
			feature_injected_vecs.append(self.inject_features(syll, syll_features))
			print('weighted and feature injected vector representation', feature_injected_vecs[-1].shape)
		word_encoding = []
		print('encoding vectors')
		for syll in feature_injected_vecs:
			word_encoding.append(self.encode(syll))
			print('vector encoded', word_encoding[-1].shape)
		print('starting bi-directional recurrent network')
		output = self.rnn_forward(torch.stack(word_encoding))
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
		prev = torch.ones((self.encode_hidden.shape[0],))
		n = 0
		length =  int(len(syll) / self.encode_in.shape[1])
		print('passing through vector')
		for i in range(length):
			input = syll[n: n + self.encode_in.shape[1]]
			print('reading context', input.shape)
			hidden = self.tanh(torch.matmul(self.encode_in, input) + self.encode_in_bias + torch.matmul(self.encode_hidden, prev) + self.encode_hidden_bias)
			prev = hidden
			n += self.encode_in.shape[1]
		return prev
			
		
		
	def embed(self, input: Tensor):
		'''
		embeds the input tensor into a compact representation, we first add a channel since the input shape is channelless
		input shape: (30,000)
		output shape: (990)
		'''
		channel = input.reshape((1,input.shape[0]))
		conv_first = self.tanh(self.conv1(channel))
		conv_second = self.tanh(self.conv2(conv_first))
		conv_third = self.tanh(self.conv3(conv_second))
		
		return conv_third.reshape(-1)
		
	def inject_features(self, embedding: Tensor, features: Tensor):
		'''
		injects our linguistic features into the embedding, then computes self attention weights with respect to the sequence and returns the weighted sum. 
		input shape: (990)
		output shape: (990)
		'''
		injected_list = []
		for feature in features:
			injected_list.append((embedding + feature))
		attended = self.attend(torch.stack(injected_list))
		return attended
		
		
	
		
	def attend(self, embedded_tensors: Tensor):
		'''
		computes the attention score of each element in the list with respect to the other elements, then returns the weighted sum of the whole
		input shape: list
		output shape: (990)
		'''
		print('starting attention network')
		weights_matrix = []
		print('compatibility is computed over', embedded_tensors.shape)
		for attention_source in embedded_tensors:
			weight_list = []
			for attention_target in embedded_tensors:
				weight_list.append(self.attention_forward(attention_source, attention_target))
				print('attention score', weight_list[-1])
			weight_tensor = torch.stack(weight_list).reshape(-1)
			print('compatibility scores computed', weight_tensor)
			weights_matrix.append(self.softmax(weight_tensor))
			print('weight distribution', weights_matrix[-1])
		weights_tensor = torch.stack((weights_matrix), dim=1)
		print('weights matrix', weights_tensor)
		attention_vector = torch.max(weights_tensor, dim=1)[0]
		print('attention vector', attention_vector)
		self.feature_weights.append(tuple(torch.round(attention_vector, decimals=2).tolist()))
		weighted = torch.matmul(attention_vector, embedded_tensors)
		return weighted
		
	
	def attention_forward(self, attention_source: Tensor, attention_target: Tensor):
		'''
		computes the compatibility score of the source to the target
		input shape: (990 * 2)
		output shape: (1)
		'''
		input = torch.cat((attention_source, attention_target))
		first_layer = self.tanh(torch.matmul( self.attnlayer1, input ) + self.attnlayer1_bias)
		second_layer = self.sigmoid(torch.matmul(self.attnlayer2, first_layer ) + self.attnlayer2_bias)
		return second_layer
		
	def rnn_forward(self, injected_list: Tensor):
		'''
		passes the list of feature injected and attention weighted tensors through one pass of a bi-directional reccurrent network, and returns the output sequence as a list of probability distributions over the two classes. 
		input shape: (n, 500)
		output shape: (n, 2)
		'''
		prev = torch.ones((self.recurrent_left_hidden.shape[0]),)
		hidden_list = []
		for input in injected_list:
			hidden = self.tanh(torch.matmul(self.recurrent_left_in, input) + self.recurrent_left_in_bias + torch.matmul(self.recurrent_left_hidden, prev) + self.recurrent_left_hidden_bias)
			hidden_list.append(hidden)
			prev = hidden
		reverse_hidden_list = []
		prev = torch.ones((self.recurrent_left_hidden.shape[0]),)
		n = len(injected_list) - 1
		while n >= 0:
			input = injected_list[n]
			hidden = self.tanh(torch.matmul(self.recurrent_right_in, input) + self.recurrent_right_in_bias + torch.matmul(self.recurrent_right_hidden, prev) + self.recurrent_right_hidden_bias)
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
			full_context.append(full)
		output_list = []
		for hidden in full_context:
			first_layer = self.sigmoid(torch.matmul(self.recurrent_out1, hidden) + self.recurrent_out1_bias)
			second_layer = self.softmax(torch.matmul(self.recurrent_out2, first_layer) + self.recurrent_out2_bias)
			output_list.append(second_layer)
		return torch.stack(output_list)
			
		
	
			
			
			
		
				
		

	
	
	