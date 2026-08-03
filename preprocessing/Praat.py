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


def combine(table: str, target: str, *sources):
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
				syllable = torch.load(word[j])
				if type(syllable) == float:
					syllable = torch.tensor(syllable).reshape((1,)) * 1000
				combinedvector.append(syllable)
			output.append(torch.cat(combinedvector))
		for k in range(len(output)):
			filename = f'{i}_{k + 1}.pt'
			torch.save(output[k], os.path.join(target, filename))
			print(target, filename, output[k].shape)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("table")
	parser.add_argument("target")
	parser.add_argument("sources", nargs='*')
	args = parser.parse_args()
	combine(Path(args.table), Path(args.target), *[Path(i) for i in args.sources])
