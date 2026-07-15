'''
the following python program transforms the input directory of wav files into pt files and saves them to the target directory
'''

from pathlib import Path
import os
import torchaudio
import torch
import torchcodec
import argparse
import re

def wav2pt(source: str, target: str):
	'''
	for each wav file in source, loads it in and writes the pt extension to the target directory
	'''
	for file in source.iterdir():
		waveform, samplerate = torchaudio.load(file)
		print(file)
		filename = os.path.split(file)[-1]
		filename = rename(filename, '.pt')
		print(target, filename)
		torch.save(waveform.reshape(-1), os.path.join(target, filename))


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
	wav2pt(Path(args.source), Path(args.target))
