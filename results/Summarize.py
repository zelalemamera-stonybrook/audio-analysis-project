'''
The following program summarizes the results in the statistics tables for presentation.
'''


import argparse
from pathlib import Path
import pandas as pd

def summarize(target: str, *tables):
	'''
	goes through each table and generates a summary of the statistics in the table. The result is written as one row in a table saved to target.
	'''
	if target.exists():
		targetstream = target.open(mode='a')
	else:
		targetstream = target.open(mode='w')
		targetstream.write('data\tmean_precision\tmax_precision\tmean_recall\tmax_recall\tmean_f\tmax_f\n')
	for path in tables:
		table = pd.read_csv(path)
		name = table['data'][0]
		precision = table['precision']
		meanp, maxp = round(float(precision.mean()), 4), round(float(precision.max()), 4)
		sensitivity = table['recall']
		means, maxs = round(float(sensitivity.mean()), 4), round(float(sensitivity.max()), 4)
		fscore = table['fscore']
		meanf, maxf = round(float(fscore.mean()), 4), round(float(fscore.max()), 4)
		targetstream.write(f'{name}\t{meanp}\t{maxp}\t{means}\t{maxs}\t{meanf}\t{maxf}\n')




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("target")
	parser.add_argument("tables", nargs='*')
	args = parser.parse_args()
	summarize(Path(args.target), *[Path(t) for t in args.tables])
