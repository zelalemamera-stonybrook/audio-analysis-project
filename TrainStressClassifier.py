'''
this program controls the training of the StressClassifier neural network. our inputs are assumed to be in the required form. we load in the data and optimize the network using SGD algorithm. 
'''
import StressClassifier


def train(network: Network , input: list[Tensor], gold: list[Tensor]):
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
	
if __name__ == '__main__':
	network = StressClassifier.Network()
	input, gold = readindata()
	train(network, input, gold)
	
	