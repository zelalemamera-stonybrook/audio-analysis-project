'''
The following program balances the training data specifically. Direct oversampling of minority classes is used.
In order to preserve disk space, actual data is not duplicated. A 'virtual' directory is used where the training algorithms passing over the data look up each datapoint repeatedly by accessing
the duplicate indices of the table.
'''

import argparse
import pandas as pd
import re
from pathlib import Path
import os

def balance(source: str, target: str):
	'''
	oversamples the table located at source by the keys in the column stress, then saves the result to target.
	'''
	print('balancing classes')
	table = pd.read_csv(source)
	table = table.set_index('Unnamed: 0')

	key = table['stress']
	counts = {}
	for i in key:
		if i in counts.keys():
			counts[i] += 1
		else:
			counts[i] = 1
	totals = []
	for key, value in counts.items():
		totals.append(value)
	print('class distribution', counts)
	majority = max(totals)
	remainders = []
	for i in counts.keys():
		idata = table[table['stress'] == i]
		print('oversampling ', majority - len(idata), 'from class', i)
		remainder = idata.sample(n = majority - len(idata), replace=True)
		remainders.append(remainder)
	remainders.append(table)
	balanced = pd.concat(remainders)
	balanced = balanced.sample(frac=1)
	print('balanced classes')
	balanced.to_csv(os.path.join(target, Path('balanced.csv')))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	balance(Path(args.source), Path(args.target))
