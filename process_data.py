import pandas as pd
import re
import shutil

train_dataframe = pd.read_csv("/Users/zelalem/Documents/LIN_487/Jordanian_data_3_syllables/Jordanian_dataset_3_syllable_train.csv")
dev_dataframe = pd.read_csv("/Users/zelalem/Documents/LIN_487/Jordanian_data_3_syllables/Jordanian_dataset_3_syllable_dev.csv")
test_dataframe = pd.read_csv("/Users/zelalem/Documents/LIN_487/Jordanian_data_3_syllables/Jordanian_dataset_3_syllable_test.csv")

if False:
    with open('/Users/zelalem/Documents/MFA/arabic_mfa.dict', 'w') as f:
        for i in range(5):
            text = dataframe['text'][i]
            ipa = dataframe['ipa'][i]
            line = f"{text}  {ipa}\n"
            f.write(line)

def write_ipa_tokens(ipa_set):
    '''
    '''
    with open('/Users/zelalem/Documents/MFA/praat/ipa.txt', 'w') as f:
        for ipa in ipa_set:
            tokens = list(ipa)
            padded = " ".join(tokens)
            f.write(f"{padded}\n")

if False:
    ipa_set = set(list(train_dataframe['ipa']))
    ipa_set = ipa_set.union(set(list(dev_dataframe['ipa'])))
    ipa_set = ipa_set.union(set(list(test_dataframe['ipa'])))
    write_ipa_tokens(ipa_set)

def remove_certain_symbols():
    with open('/Users/zelalem/Documents/MFA/praat/ipa.txt', 'r') as f:
        new_token_list = []
        for line in f:
            cleaned_list = []
            token_list = re.split(r'\s', line)
            bad_set = set(['ˈ', '.', ')', '('])
            for token in token_list:
                if token not in bad_set:
                    cleaned_list.append(token)
            new_token_list.append(" ".join(cleaned_list))
        with open('/Users/zelalem/Documents/MFA/praat/ipa_without_suprasegmentals.txt', 'w') as g:
            for token in new_token_list:
                g.write(f"{token}\n")

if False:
    remove_certain_symbols()

forbidden_list = ['ˌ', '‿', ' ˈ']

suprasegmental_set = set(['ˤ', 'ː'])

if False:
    with open('/Users/zelalem/Documents/MFA/praat/ipa_without_suprasegmentals.txt', 'r') as f:
        new_lines = []
        for line in f:
            token_list = re.split(r"\s", line)
            new_token_list = []
            for i, token in enumerate(token_list):
                if token not in suprasegmental_set:
                    new_token_list.append(token)
                else:
                    char = f"{new_token_list[-1]}{token}"
                    new_token_list[-1] = char
            new_lines.append(" ".join(new_token_list))
        with open('/Users/zelalem/Documents/MFA/praat/ipa_without_suprasegmentals.txt', 'w') as g:
            for line in new_lines:
                g.write(f"{line}\n")

if False:
    token_set = set()
    with open('/Users/zelalem/Documents/MFA/praat/ipa_without_suprasegmentals.txt', 'r') as f:
        for line in f:
            line = line.strip()
            char_list = re.split(r'\s', line)
            char_set = set(char_list)
            token_set = token_set.union(char_set)
    with open('/Users/zelalem/Documents/MFA/praat/ipa_symbols.txt', 'w') as f:
        for char in token_set:
            f.write(f"{char}\n")

if False:
    ipa_set = set()
    model_set = set()
    with open('/Users/zelalem/Documents/MFA/praat/ipa_symbols.txt', 'r') as f:
        for char in f:
            char = char.strip()
            ipa_set.add(char)
    with open('/Users/zelalem/Documents/MFA/praat/model_symbols.txt', 'r') as f:
        for char in f:
            char = char.strip()
            model_set.add(char)
    intersection = ipa_set.intersection(model_set)
    with open('/Users/zelalem/Documents/MFA/praat/ipa_to_model_symbols.txt', 'w') as f:
        f.write("ipa   model\n")
        for char in intersection:
            line = f"{char}\t{char}\n"
            f.write(line)
        still_missing = ipa_set.difference(intersection)
        for char in still_missing:
            line = f"{char} \n"
            f.write(line)
def process_ipa(ipa, ipa_to_model_dict):
    '''
    '''
    tokens = list(ipa)
    bad_list =['ˈ', ')', '.', '(']
    new_ipa = []
    for i, char in enumerate(tokens):
        if char == bad_list[0] and i > 0:
            tokens[i] = bad_list[2]
    for char in tokens:
        if char not in bad_list or char == bad_list[2]:
            new_ipa.append(char)
    syllable_index = []
    suprasegmental_set = set(['ˤ', 'ː'])
    new_token_list = []
    for i, token in enumerate(new_ipa):
        if token not in suprasegmental_set:
            new_token_list.append(token)
        else:
            char = f"{new_token_list[-1]}{token}"
            new_token_list[-1] = char
    for i, token in enumerate(new_token_list):
        if token == bad_list[2]:
            syllable_index.append(i)
    syllable_index[1] -= 1
    final_ipa = []
    for token in new_token_list:
        if token not in bad_list:
            final_ipa.append(token)
    model_ipa = []
    for token in final_ipa:
        model_ipa.append(ipa_to_model_dict[token])
    return " ".join(model_ipa), syllable_index
  

if False:
    ipa_to_model_dictionary = dict()
    with open('Users/zelalem/Documents/MFA/praat/ipa_to_model_symbols.txt', 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            line = line.strip()
            x_y = re.split('\t', line)
            ipa_to_model_dictionary[x_y[0]] = x_y[1]
    to_be_written = [process_ipa(ipa, ipa_to_model_dictionary) for ipa in train_dataframe['ipa']]
    model_ipa = [tup[0] for tup in to_be_written]
    syllable_locations = [tup[1] for tup in to_be_written]
    arabic_dictionary = {orthography:model_ipa for orthography, model_ipa in zip(train_dataframe['text'], model_ipa)}
    print(arabic_dictionary['اختصر'])
    if True:
        with open('Users/zelalem/Documents/MFA/praat/arabic_mfa.dict', 'w') as f:
            for key, value in arabic_dictionary.items():
                key = key.strip()
                value = value.strip()
                f.write(f"{key}\t{value}\n")
    if True:
        with open('Users/zelalem/Documents/MFA/praat/syllablelocations.txt', 'w') as f:
            f.write("filename\tsyll1\tsyll2\n")
            for i, syllist in enumerate(syllable_locations):
                line = f"file{i}.TextGrid\t{syllist[0]}\t{syllist[1]}\n"
                f.write(line)
    wavnames =[re.findall(r"LL-Q55633582.*", text)[0] for text in train_dataframe['audio_urls']]
    if True:
        for i, wavname in enumerate(wavnames):
            input_path = f"/Users/zelalem/Documents/LIN_487/south_levantine_data/ajp/{wavname}"
            target_path = f"/Users/zelalem/Documents/MFA/my_corpus/train/file{i}.wav"
            target_path2 = f"/Users/zelalem/Documents/MFA/praat/wavfiles/file{i}.wav"
            shutil.copy(input_path, target_path)
            shutil.copy(input_path, target_path2)
    arabic_orthography = [orthography for orthography in train_dataframe['text']]
    if True:
        for i, orthography in enumerate(arabic_orthography):
            file_path = f"/Users/zelalem/Documents/MFA/my_corpus/train/file{i}.txt"
            orthography = orthography.strip()
            with open(file_path, 'w') as f:
                f.write(orthography)

if True:
    wavnames =[re.findall(r"LL-Q55633582.*", text)[0] for text in dev_dataframe['audio_urls']]
    for i, wavname in enumerate(wavnames):
        input_path = f"/Users/zelalem/Documents/LIN_487/south_levantine_data/ajp/{wavname}"
        target_path = f"/Users/zelalem/Documents/MFA/my_corpus/dev/file{i}.wav"
        target_path2 = f"/Users/zelalem/Documents/MFA/praat/wavfiles/dev/file{i}.wav"
        shutil.copy(input_path, target_path)
        shutil.copy(input_path, target_path2)
    ipa_to_model_dictionary = dict()
    with open('/Users/zelalem/Documents/MFA/praat/ipa_to_model_symbols.txt', 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            line = line.strip()
            x_y = re.split('\t', line)
            ipa_to_model_dictionary[x_y[0]] = x_y[1]
    to_be_written = [process_ipa(ipa, ipa_to_model_dictionary) for ipa in dev_dataframe['ipa']]
    model_ipa = [tup[0] for tup in to_be_written]
    syllable_locations = [tup[1] for tup in to_be_written]
    arabic_dictionary = {orthography:model_ipa for orthography, model_ipa in zip(dev_dataframe['text'], model_ipa)}
    with open('/Users/zelalem/Documents/MFA/praat/arabic_mfa.dict', 'w') as f:
        for key, value in arabic_dictionary.items():
            key = key.strip()
            value = value.strip()
            f.write(f"{key}\t{value}\n")
    with open('/Users/zelalem/Documents/MFA/praat/syllablelocations.txt', 'w') as f:
            f.write("filename\tsyll1\tsyll2\n")
            for i, syllist in enumerate(syllable_locations):
                line = f"file{i}.TextGrid\t{syllist[0]}\t{syllist[1]}\n"
                f.write(line)
    arabic_orthography = [orthography for orthography in dev_dataframe['text']]
    for i, orthography in enumerate(arabic_orthography):
            file_path = f"/Users/zelalem/Documents/MFA/my_corpus/dev/file{i}.txt"
            orthography = orthography.strip()
            with open(file_path, 'w') as f:
                f.write(orthography)
    
    
