'''
The following program normalizes the input directory. The directory is assumed to be populated with one dimensional vectors of the same length. This aligorithm
normalizes along each individual dimension.
'''

import argparse
from pathlib import Path
import torch
import os

def normalize(source: str, target: str):
	'''
	computes the center of the data for every dimension, then subtracts each dimension from it. This method does not scale the values by standard deviation.
	'''
	dim, size = collect_statistics(source)
	total = torch.zeros((dim,))
	for file in source.iterdir():
		vector = torch.load(file)
		total += vector
	mean = total / size
	for file in source.iterdir():
		vector = torch.load(file)
		normal = vector - mean
		filename = os.path.split(file)[-1]
		torch.save(normal, os.path.join(target, filename))


def collect_statistics(source: str):
	'''
	returns the dimension of each vector and the total size of the source directory
	'''
	file = next(source.glob('*'))
	vector = torch.load(file)
	dim = len(vector)
	size = len(list(source.glob('*')))
	return dim, size



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	normalize(Path(args.source), Path(args.target))
