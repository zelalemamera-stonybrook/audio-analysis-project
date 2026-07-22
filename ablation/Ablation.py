'''
the following python script takes as its input a list of directories with praat generated tensors and will produce a new directory which for each word in the source directories
combines its different representations into one fixed tensor. Since the size of a vector over a fixed directory is constant throughout, this process will produce a directory with also a constant size
throughout. Hence no padding is needed.
'''
import argparse
import torch
from pathlib import Path
import pandas as pd
import os


def combine(table: str, target: str, omit: str,  *sources):
	'''
	uses the index in the table to find each word in sources and combine it, then saves the result to target.
	'''
	table = pd.read_csv(table)
	table = table.set_index('Unnamed: 0')
	for i in table.index:
		output = []
		combined = []
		wordsize = 0
		for source in sources:
			word = sorted(list(source.glob(f'{i}_*')))
			combined.append(word)
			print(source, word)
		wordsize = len(combined[-1])
		for j in range(wordsize):
			combinedvector = []
			for word in combined:
				syllable = word[j]
				prefix = os.path.split(syllable)[0]
				if Path(prefix) == omit:
					print('ommited', syllable)
					syllable = torch.load(syllable)
					if type(syllable) == float:
						syllable = torch.tensor([0])
					else:
						syllable = torch.zeros(syllable.shape)
				else:
					syllable = torch.load(syllable)
					if type(syllable) == float:
						syllable = torch.tensor(syllable).reshape((1,))
				combinedvector.append(syllable)
			output.append(torch.cat(combinedvector))
		for k in range(len(output)):
			filename = f'{i}_{k + 1}.pt'
			torch.save(output[k], os.path.join(target, filename))
			print(target, filename, output[k].shape)

def getfeaturelist(source: str):
	'''
	gets every feature directory in source.
	'''
	featurelist = ['Dur', 'F0norm', 'F1norm', 'F2norm', 'F3norm', 'F4norm', 'F5norm', 'Intensitynorm']
	pathlist = []
	for f in featurelist:
		path = os.path.join(source, f)
		pathlist.append(path)
	return pathlist


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("table")
	parser.add_argument("target")
	parser.add_argument("source")
	parser.add_argument("omit", default=None)
	args = parser.parse_args()
	sources = getfeaturelist(Path(args.source))
	combine(Path(args.table), Path(args.target), Path(args.omit), *[Path(i) for i in sources])
