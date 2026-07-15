'''
the following program transforms the pitch directory which is made out of .csv files, into vectors that can be used directly by the model
these will be stored in the F0 folder as .pt files
'''
import argparse
from pathlib import Path
import pandas as pd
import torch
import os
import re


def transform_tables(source: str, target: str):
	'''
	reads every csv table in the source directory, each table is assumed to contain exactly one row with the pitch information contained over time sections.
	this row is turned into a vector and stored as a pt file of the same name
	'''
	for file in source.iterdir():
		print(file)
		table = pd.read_csv(file)
		table = table.drop(columns='none')
		vector = torch.tensor(table.to_numpy().tolist()[0])
		filename = os.path.split(file)[-1]
		filename = rename(filename, '.pt')
		print(target, filename)
		torch.save(vector, os.path.join(target, filename))

def rename(filename: str, extension: str):
        '''
        uses a regular pattern to rename filename's extension to the specified ending
        '''
        raw_name = re.split(r'\.', str(filename))[0]
        return Path(f'{raw_name}{extension}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	transform_tables(Path(args.source), Path(args.target))
