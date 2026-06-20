'''
The following program prepares audio files for forced alignment. This is the first step of the data processing pipeline that feeds into the StressClassifier neural network. After this program runs, the relelant audio and text directories have been generated for the alignment. 
We use montreal forced aligner to obtain the aligned corpus and proceed to the next step of the pipeline. The user is ultimately expected to generate the appropriate syllabifications via the Praat software. All of the data necessary to do this will be generated.
'''

from pathlib import Path
import json
import re
import pandas as pd
from pandas import DataFrame
import shutil
import os
import subprocess

path = Path('data/ipa_to_mfa.json')
model_symbols = {}
with path.open(mode='r') as f:
		model_symbols = json.load(f)

def prepare_alignments():
	'''
	the data is assumed to be written into exactly three tables. one containing two sullable data, another containing three syllable data, and lastly one contaiining four syllable data.
	each table will be read from the subdirectory called data, null values will be dropped, tables will be split randomly into 80/10/10 protions representing the training, testing, and development samples respectively,
	then columns are processed one by one for each sample; ipa text will be processed for generating index locations, and text column will be used to 
	generate dictionaries for the alignment, audio paths will be checked for validity before they are batched for alignment. all of the audio files needed are assumed to be present somewhere in the current directory under the audio subfolder.
	once the nessary data for alignment has been generated, the program is terminated. each batch should be subsequently aligned using Montreal Forced Aligner from the command line.
	'''
	print('reading in tables')
	data_2 = pd.read_csv(Path('./data/data_2.csv'))
	data_3 = pd.read_csv(Path('./data/data_3.csv'))
	data_4 = pd.read_csv(Path('./data/data_4.csv'))
	print('sucessfuly read tables')
	data = {'data_2': data_2, 'data_3': data_3, 'data_4': data_4}
	split = split_data(data)
	generate_text(split)
	generate_audio(split)
	generate_aligner_dictionary(data)
	
def split_data(data: dict):
	'''
	80/10/10 split
	'''
	print('splitting data')
	split = {}
	for key, val in data.items():
		train = val.sample(frac = .8)
		remainder = val.drop(list(train.index))
		test = remainder.sample(frac = .5)
		dev = remainder.drop(list(test.index))
		split[f'{key}'] = {}
		
		split[f'{key}']['train'] = train
		trainpath = f'data/{key}/train/train.csv'
		subprocess.run(['rm', trainpath])
		print('saving to', trainpath)
		train.to_csv(trainpath)
		
		split[f'{key}']['test'] = test
		testpath = f'data/{key}/test/test.csv'
		subprocess.run(['rm', testpath])
		print('saving to', testpath)
		test.to_csv(testpath)
		
		split[f'{key}']['dev'] = dev
		devpath = f'data/{key}/dev/dev.csv'
		subprocess.run(['rm', devpath])
		print('saving to', devpath)
		dev.to_csv(devpath)
		
	print('sucessfully split data', split.items())
	return split
	
def generate_text(split: dict):
	'''
	each table contains a text column which will be used to align the audio file, this text column is used to populate the text directory of the alignment.
	'''
	print('generating text')
	for key, subdict in split.items():
		for batch, table in subdict.items():
			address = f'./data/{key}/{batch}/alignment/text'
			shutil.rmtree(address)
			os.mkdir(address)
			for i in table.index:
				write_text_file(key, batch, i, table['text'][i])
	print('generated text files')
				
def write_text_file(key: str, batch: str, i: int, text: str):
	'''
	writes text to the directory data/key/batch/alignment/text/file_i.txt
	'''
	path = Path(f'./data/{key}/{batch}/alignment/text/file_{i}.txt')
	with path.open(mode='w') as f:
		f.write(remove_whitespace(text))
		
def remove_whitespace(text: str):
	'''
	removes whitespace
	'''
	whitespace = re.compile(r'\s')
	return whitespace.sub("", text)
	
def generate_audio(split: dict):
	'''
	each table contains a column locating an audio file for that word, this is copied from its location and used to populatte the audio folder of the alignment 
	'''
	print('generating audio files')
	for key, subdict in split.items():
		for batch, table in subdict.items():
			shutil.rmtree(f'data/{key}/{batch}/alignment/audio')
			os.mkdir(f'data/{key}/{batch}/alignment/audio')
			for i in table.index:
				copy_audio_file(key, batch, i, table['audio_urls'][i])
	print('generated audio files')
				
				
def copy_audio_file(key: str, batch: str, i: int, address: str):
	'''
	copies the file located at data/audio/address to data/key/batch/alignment/audio/file_i.wav
	'''
	print('current address', address, 'for file number', i, 'in', key, batch)
	address = filter_audio_url(address)
	source = f'data/audio/{address}'
	target = f'data/{key}/{batch}/alignment/audio/file_{i}.wav'
	shutil.copy(source, target)
	print('audio file copied successfuly')
	
def filter_audio_url(address: str):
	'''
	returns the hashable location of audio
	'''
	return re.search(r"LL-Q55633582.*?wav", address).group()

def generate_aligner_dictionary(data: dict):
	'''
	montreal requires a dictionary of mappings from the arabic directly to a phoneme sequence in the vocabulary providded by the acoustic model of MFA. currently, there is no such dictionary
	so we will generate one from the data by mapping the arabic text directly to its already provided ipa trasncription and then mapping this to the character set of the phonemes in MFA.
	the function that maps the observed chracters in the data to phonemes in MFA is already created and stored in the directory. 
	'''
	print('generating aligner vocabulary')
	for key, value in data.items():
		vocab = zip(value['text'], value['ipa'])
		dictionary = {clean_text(txt):map_to_mfa(ipa) for txt, ipa in vocab}
		write_model_dictionary(key, dictionary)
		
def clean_text(txt: str):
	'''
	processes the text part of data
	'''
	print('received text', txt)
	val = re.sub(r"\s", "", txt)
	print('writing ', val, 'in dict')
	return val
	
def write_model_dictionary(key: str, dictionary: dict):
	'''
	writes the model dictionary to data/key/arabic_mfa.dict
	'''
	path = Path(f'data/{key}/arabic_mfa.dict')
	with path.open(mode='w') as f:
		for key, value in dictionary.items():
			line = f'{key}\t{value}\n'
			f.write(line)
			
			
def map_to_mfa(ipa: str):
	'''
	the raw ipa strings are not sutiatble for alignment before putting them into a specific format; any suprasegmentals need to be removed if they are not a part of mfa vocabulary,
	additionaly, mfa strings are delimited by space, however the ipa provided does not come with space delimitation, this needs to be added with the additional comlplication of that any suprasegmatal sequences attatched to an 
	ipa character should be considered a part of the same token. after this cleaning is applied, the ipa string is sent to the raw ipa to model symbols function to be trnasformed into an roughly equivalent sequence of an mfa phone string, 
	which can be used for alignment directly.
	'''
	print('received ipa', ipa)
	ipa = re.sub(r"\s", "", ipa)
	unwanted = set([' ', ')', '(', '‿', 'ˌ', 'ˈ', '.', '͡'])
	suprasegmentals = set(['ˤ', 'ː'])
	ipa_list = list(ipa)
	cleaned_list = []
	segmented_list = []
	for ipa in ipa_list:
		if ipa not in unwanted:
			cleaned_list.append(ipa)
	print('cleaned version of ipa', cleaned_list)
	for i, ipa in enumerate(cleaned_list):
		if ipa in suprasegmentals:
			segmented_list[-1] = f"{cleaned_list[i - 1]}{ipa}"
		else:
			segmented_list.append(ipa)
	for i, ipa in enumerate(segmented_list):
		print('mapping', segmented_list[i], 'to', model_symbols[ipa])
		segmented_list[i] = model_symbols[ipa]
	return " ".join(segmented_list)
			

if __name__ == '__main__':
	prepare_alignments()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	