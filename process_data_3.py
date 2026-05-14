import pandas as pd
import re
import numpy as np
import os
import torch
import shutil

df = pd.read_csv('/Users/zelalem/Documents/LIN_487/south_levantine_data/south_levantine_wiktionary_ipa.csv')

text = [text for text in df['page_title']]
ipa = [ipa for ipa in df['ipa']]
audio_files = [files for files in df['audio_files']]
audio_urls = [urls for urls in df['audio_urls']]

def clean_ipa(text):
    '''
    takes the ipa and produces a clean version
    '''
    ipa_list = re.findall(r"(<p:.*?>)", text)
    new_list = [re.sub(r'[(<p:)>\s]', '', txt) for txt in ipa_list]
    if len(new_list) == 0:
        return ''
    else:
        return new_list[0]

def syllabify(ipa):
    '''
    returns the numpber of syllables of this ipa
    '''
    stress_marker = 'ˈ'
    if len(ipa) == 0:
        return []
    substrings = re.split(r'\.', ipa)
    syllable_list = []
    for substring in substrings:
        if substring[0] == stress_marker and substring[-1] != stress_marker:
            syllable_list += re.split(f"{stress_marker}", substring)[1:]
        elif substring[-1] == stress_marker and substring[0] != stress_marker:
            syllable_list += re.split(f"{stress_marker}", substring)[0:-1]
        elif substring[0] == substring[-1] and substring[0] == stress_marker:
            syllable_list += re.split(f"{stress_marker}", substring)[1:-1]
        else:
            syllable_list += re.split(f"{stress_marker}", substring)
    return syllable_list

cleaned_ipa = [clean_ipa(txt) for txt in ipa]
syllables = [len(syllabify(txt)) for txt in cleaned_ipa]
dataframe = pd.DataFrame({'text':text, 'ipa':cleaned_ipa, 'syllables':syllables, 'audio_files':audio_files, 'audio_urls':audio_urls}, index=range(len(text)))
    
def check_none(obj):
    '''
    '''
    if obj:
        return True
    else:
        return False
    
def generate_one_syllables(dataframe):
    '''
    the dataframe is assumed to contain all of the files, this returns only the one syllable files and they will be cleaned
    '''
    df = dataframe
    temp = df[df['syllables'] == 1]
    temp = temp[[type(i) != float for i in temp['audio_urls']]]
    temp = temp[[len(re.findall(r'ˈ', text)) == 0 for text in temp['ipa']]]
    temp = temp[[len(re.split(r"\s", txt.strip())) == 1 for txt in temp['ipa']]]
    temp = temp[[check_none(re.search("LL-Q55633582.*?wav", urls)) for urls in temp['audio_urls']]]
    audio_urls = [re.search("LL-Q55633582.*?wav", urls).group() for urls in temp['audio_urls']]
    temp = temp.drop(columns='audio_urls')
    temp['audio_urls'] = audio_urls
    temp = temp[[os.path.isfile(f"/Users/zelalem/Documents/LIN_487/south_levantine_data/ajp/{link}") for link in temp['audio_urls']]]
    set_of_bad = set(['bænædɔːɾɑ', 'bxsˤuːsˤ', 'kɨtæːb', 'ləbnæːn'])
    temp = temp[[ipa not in set_of_bad for ipa in temp['ipa']]]
    stress = torch.ones(len(temp))
    stress = stress.tolist()
    temp['stress'] = stress
    temp.to_csv('/Users/zelalem/Documents/LIN_487/one_syllable_data.csv')
    return temp

def generate_four_syllables(dataframe):
    '''
    generates the clean set of four syllable words to be used
    '''
    df = dataframe
    temp = df[df['syllables'] == 4]
    temp = temp.dropna()
    stress = [count_stress(txt) for txt in temp['ipa']]
    temp['stress'] = stress
    audio_urls = [re.search("LL-Q55633582.*?wav", urls).group() for urls in temp['audio_urls']]
    temp = temp.drop(columns='audio_urls')
    temp['audio_urls'] = audio_urls
    temp.to_csv('/Users/zelalem/Documents/LIN_487/four_syllable_data.csv')
    return temp

def count_stress(ipa):
    '''
    returns the location of the stress marker in this ipa. one is assumed to exist.
    '''
    syll_list = syllabify(ipa)
    stress_marker = 'ˈ'
    for i, syll in enumerate(syll_list):
        if re.search(f'\.?{stress_marker}\.?{syll}', ipa):
            return i + 1
    return -1

def make_traindevtest_split(dataframe):
    '''
    resamples an 80/10/10 from this dataframe and returns them as train, dev, test
    '''
    train = dataframe.sample(frac = .8)
    remainder = dataframe.drop(list(train.index))
    test = remainder.sample(frac = .5)
    dev = remainder.drop(list(test.index))
    return train, dev, test

def save_splits_tofolder(dataframe, path):
    '''
    saves this dataframe to the path
    '''
    dataframe.to_csv(path)

def write():
    '''
    automater function
    '''
    list_of_frames = [generate_one_syllables(dataframe), generate_four_syllables(dataframe)]
    path = '/Users/zelalem/Documents/LIN_487/'
    list_of_strings = [f"{path}one_syllable_data/one_syllable_data", f"{path}four_syllable_data/four_syllable_data"]
    list_of_names = ['_train.csv', '_dev.csv', '_test.csv']

    for df, path in zip(list_of_frames, list_of_strings):
        train, dev, test = make_traindevtest_split(df)
        for df, name in zip([train, dev, test], list_of_names):
            save_splits_tofolder(df, f"{path}{name}")

def generate_alignment_corpus(dataframe, path):
    '''
    populates this folder with wav file and txt file pairs
    '''
    text = [re.sub('\s', '', txt) for txt in dataframe['text']]
    names = [f"file{i}" for i, text in enumerate(text)]
    wav_directory = '/Users/zelalem/Documents/LIN_487/south_levantine_data/ajp/'
    for name, location in zip(names, dataframe['audio_urls']):
        shutil.copy(f"{wav_directory}{location}", f"{path}{name}.wav")
    for name, text in zip(names, text):
        with open(f'{path}{name}.txt', 'w') as f:
            f.write(text)

def write1():
    '''
    automater function
    '''
    path = '/Users/zelalem/Documents/LIN_487/'
    target = '/Users/zelalem/Documents/my_corpus/'

    location_list = [f'{path}one_syllable_data/one_syllable_data', f'{path}four_syllable_data/four_syllable_data']
    target_list = [f'{target}syllable_1/', f'{target}syllable_4/']

    list_of_types = ['train/', 'dev/', 'test/']
    list_of_names = ['_train.csv', '_dev.csv', '_test.csv']

    for i, location in enumerate(location_list):
        df_list = []
        for name in list_of_names:
            df_list.append(pd.read_csv(f'{location}{name}'))
        for df, name in zip(df_list, list_of_types):
            target = target_list[i]
            generate_alignment_corpus(df, f'{target}{name}')

def clean_ipa(ipa):
    '''
    removes the undesirable list and concatenates any suprasegmentals, returns the cleaned ipa padded by space
    '''
    unwanted = set([' ', ')', '(', '‿', 'ˌ', 'ˈ', '.', '͡'])
    suprasegmentals = set(['ˤ', 'ː'])
    ipa_list = list(ipa)
    cleaned_list = []
    segmented_list = []
    for ipa in ipa_list:
        if ipa not in unwanted:
            cleaned_list.append(ipa)
    for i, ipa in enumerate(cleaned_list):
        if ipa in suprasegmentals:
            segmented_list[-1] = f"{cleaned_list[i - 1]}{ipa}"
        else:
            segmented_list.append(ipa)
    for i, ipa in enumerate(segmented_list):
        segmented_list[i] = model_symbols(ipa)
    return " ".join(segmented_list)

def model_symbols(char):
    '''
    returns the model version of char
    '''
    dct = {}
    with open('/Users/zelalem/Documents/praat/aligner_vocab/2_syllables/ipa_to_model_symbols.txt', 'r') as f:
        next(f)
        for line in f:
            str = re.split("\s", line.strip())
            ipa, model= str[0].strip(), str[-1].strip()
            dct[ipa] = model
    return dct[char]

def generate_dictionary(dataframe, path):
    '''
    writes a text to model ipa dictionary to the path
    '''
    with open(f'{path}arabic_mfa.dict', 'w') as f:
        for text, ipa in zip(dataframe['text'], dataframe['ipa']):
            text = re.sub('\s', '', text)
            model_ipa = clean_ipa(ipa)
            f.write(f'{text}\t{model_ipa}\n')
    
def write2():
    '''
    automater function
    '''
    path = '/Users/zelalem/Documents/LIN_487/'
    target = '/Users/zelalem/Documents/praat/aligner_vocab/'

    location_list = [f'{path}one_syllable_data/one_syllable_data', f'{path}four_syllable_data/four_syllable_data']
    target_list = [f'{target}1_syllable/', f'{target}4_syllables/']

    list_of_types = ['train/', 'dev/', 'test/']
    list_of_names = ['_train.csv', '_dev.csv', '_test.csv']

    for i, source in enumerate(location_list):
        df_list = []
        for name in list_of_names:
            df_list.append(pd.read_csv(f'{source}{name}'))
        for df, target in zip(df_list, list_of_types):
            generate_dictionary(df, f'{target_list[i]}{target}')

def find_syllable_location(ipa):
    '''
    returns an integer representing the number of model symbols that preceede the occurence of each syllable marker
    '''
    syllables = syllabify(ipa)
    stress_list = [0]
    for syll in syllables:
        model = clean_ipa(syll)
        stress_list.append(len(re.split('\s', model)) + stress_list[-1])
    stress_list = stress_list[1:-1]
    return stress_list

def write_syllable_location(dataframe, path):
    '''
    writes the location of all the syllables to this path for this dataframe
    '''
    with open(f'{path}sylllocations.txt', 'w') as f:
        f.write('filename\tsyll1\tsyll2\tsyll3\n')
        names = [f'file{i}.TextGrid' for i, txt in enumerate(dataframe['text'])]
        sylls = [find_syllable_location(ipa) for ipa in dataframe['ipa']]
        for name, syll_list in zip(names, sylls):
            syll1 = syll_list[0]
            syll2 = syll_list[1]
            syll3 = syll_list[-1]
            f.write(f'{name}\t{syll1}\t{syll2}\t{syll3}\n')

def write3():
    '''
    automater function
    '''
    path = '/Users/zelalem/Documents/LIN_487/'
    target = '/Users/zelalem/Documents/praat/aligner_vocab/'

    location_list = [f'{path}four_syllable_data/four_syllable_data']
    target_list = [f'{target}4_syllables/']

    list_of_types = ['train/', 'dev/', 'test/']
    list_of_names = ['_train.csv', '_dev.csv', '_test.csv']

    for i, source in enumerate(location_list):
        df_list = []
        for name in list_of_names:
            df_list.append(pd.read_csv(f'{source}{name}'))
        for df, target in zip(df_list, list_of_types):
            write_syllable_location(df, f'{target_list[i]}{target}')

def generate_praat_textgrids(sourcepath, targetpath, names):
    '''
    populates the targetpath by sourcepath texgrids matching names
    '''
    for name in names:
        shutil.copy(f'{sourcepath}{name}', f'{targetpath}{name}')

def generate_praat_waveforms(sourcepath, targetpath, names):
    '''
    populates the targetpath by sourcepath wavfiles matching names
    '''
    for name in names:
        shutil.copy(f'{sourcepath}{name}', f'{targetpath}{name}')

def write4():
    '''
    '''
    sourcetext = '/Users/zelalem/Documents/my_corpus_aligned/'
    sourcewav  = '/Users/zelalem/Documents/my_corpus/'

    syllables = ['syllable_1/', 'syllable_4/']

    types = ['train/', 'dev/', 'test/']

    file_sizes = [[278, 35, 35], [36, 5, 4]]

    texttarget = '/Users/zelalem/Documents/praat/textgrids/'
    wavtarget = '/Users/zelalem/Documents/praat/wavfiles/'

    for i, syll in enumerate(syllables):
        for tp, file_size in zip(types, file_sizes[i]):
            wavnames = [f'file{i}.wav' for i in range(file_size)]
            generate_praat_waveforms(f'{sourcewav}{syll}{tp}', f'{wavtarget}{syll}{tp}', wavnames)
            textnames = [f'file{i}.TextGrid' for i in range(file_size)]
            generate_praat_textgrids(f'{sourcetext}{syll}{tp}', f'{texttarget}{syll}{tp}', textnames)

write4()




