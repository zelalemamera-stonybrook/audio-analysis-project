import pandas as pd
import re
import shutil

train_dataframe = pd.read_csv("/Users/zelalem/Documents/LIN_487/Jordanian_data_2_syllables/Jordanian_dataset_2_syllable_train.csv")
dev_dataframe = pd.read_csv("/Users/zelalem/Documents/LIN_487/Jordanian_data_2_syllables/Jordanian_dataset_2_syllable_dev.csv")
test_dataframe = pd.read_csv("/Users/zelalem/Documents/LIN_487/Jordanian_data_2_syllables/Jordanian_dataset_2_syllable_test.csv")
total_dataframe = pd.read_csv("/Users/zelalem/Documents/LIN_487/Jordanian_data_2_syllables/Jordanian_dataset_2_syllable.csv")

def write_dictionary():
    '''
    writes the dictionary for our dataset
    '''
    with open('/Users/zelalem/Documents/praat/aligner_vocab/2_syllables/arabic_mfa.dict', 'w') as f:
        textipa = [tup for tup in zip(total_dataframe['text'], total_dataframe['ipa'])]
        for text, ipa in textipa:
            clean_ip = clean_ipa(ipa)
            line = f"{text}\t{clean_ip}\n"
            f.write(line)

def clean_ipa(ipa):
    '''
    removes the undesirable list and concatenates any suprasegmentals, returns the cleaned ipa padded by space
    '''
    unwanted = set([' ', ')', '(', '‿', 'ˌ', 'ˈ', '.'])
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

def find_syllable_location(ipa):
    '''
    returns the location of the syllable marker for this ipa
    '''
    if len(re.findall(r"ˈ", ipa)) == 0:
            ipa = f"ˈ{ipa}"
    if ipa[0] == 'ˈ':
        ipa = ipa[1:]
        split_list = re.split(r'\.', ipa)
        left = clean_ipa(split_list[0])
        syllable_location = len(re.split(r"\s", left))
        return syllable_location
    else:
        split_list = re.split(r'ˈ', ipa)
        left = clean_ipa(split_list[0])
        syllable_location = len(re.split(r"\s", left))
        return syllable_location

def write_syllable_location(dataframe, path):
    '''
    logs the syllable location for each file as inputed
    '''
    with open(path, 'w') as f:
        ipa_list = [ipa for ipa in dataframe['ipa']]
        f.write("filename\tsyll1\n")
        for i, ipa in enumerate(ipa_list):
            location = find_syllable_location(ipa)
            line = f"file{i}.TextGrid\t{location}\n"
            f.write(line)

def write_all_sylllocations():
    '''
    automated function
    '''
    path_list = ['/Users/zelalem/Documents/praat/aligner_vocab/2_syllables/train/syllablelocations.txt',
                 '/Users/zelalem/Documents/praat/aligner_vocab/2_syllables/dev/syllablelocations.txt',
                 '/Users/zelalem/Documents/praat/aligner_vocab/2_syllables/test/syllablelocations.txt']
    dataframe_list = [train_dataframe, dev_dataframe, test_dataframe]
    for dataframe, path in zip(dataframe_list, path_list):
        write_syllable_location(dataframe, path)



def build_corpus(path1, path2, dataframe):
    '''
    writes the set of sound and text files to the locations for alignment
    '''
    wavnames =[re.findall(r"LL-Q55633582.*", text)[0] for text in dataframe['audio_urls']]
    print(len(wavnames))
    for i, wavname in enumerate(wavnames):
        input_path = f"/Users/zelalem/Documents/LIN_487/south_levantine_data/ajp/{wavname}"
        target_path = f"{path1}/file{i}.wav"
        target_path2 = f"{path2}/file{i}.wav"
        shutil.copy(input_path, target_path)
        shutil.copy(input_path, target_path2)
    arabic_orthography = [orthography for orthography in dataframe['text']]
    for i, orthography in enumerate(arabic_orthography):
        file_path = f"{path1}/file{i}.txt"
        orthography = orthography.strip()
        with open(file_path, 'w') as f:
            f.write(orthography)

def build_all_corpora():
    '''
    automated function
    '''
    name_list = ['train', 'test', 'dev']
    dataframe_list = [train_dataframe, test_dataframe, dev_dataframe]
    path_tuple = [(f"/Users/zelalem/Documents/my_corpus/syllable_2/{name}", f"/Users/zelalem/Documents/praat/wavfiles/syllable_2/{name}") for name in name_list]
    for dataframe, tup in zip(dataframe_list, path_tuple):
        build_corpus(tup[0], tup[1], dataframe)

write_all_sylllocations()
