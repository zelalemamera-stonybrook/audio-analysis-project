'''
The following python program generates  duration information from all of the audio provided in the directory.
The audio is assumed to contain no silence intervals on either end so that its direct length can be used in the computation.
'''

import argparse
from pathlib import Path
import re
import torchaudio
import torch
import torchcodec
import os

def saveduration(source: str, target: str):
	'''
	computes the duration of each audio in source and saves it to target. Duration is defined to be length / samplerate.
	'''
	for file in source.iterdir():
		waveform, samplerate = torchaudio.load(file)
		print(file)
		duration = len(waveform[0]) / samplerate
		filename = os.path.split(file)[-1]
		filename = rename(filename, '.pt')
		print(target, filename)
		torch.save(duration, os.path.join(target, filename))


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
	saveduration(Path(args.source), Path(args.target))
