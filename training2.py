import pandas as pd
import re
import torch

train_formant_1_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_1/train/duration_formant.csv')
train_intensity_1_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_1/train/duration_intensity.csv')
train_intensity_1_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
train_pitch_1_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_1/train/duration_pitch.csv')
train_pitch_1_syll_table.drop(columns='duration', inplace=True)

dev_formant_1_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_1/dev/duration_formant.csv')
dev_intensity_1_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_1/dev/duration_intensity.csv')
dev_intensity_1_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
dev_pitch_1_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_1/dev/duration_pitch.csv')
dev_pitch_1_syll_table.drop(columns='duration', inplace=True)

train_formant_2_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_2/train/duration_formant.csv')
train_intensity_2_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_2/train/duration_intensity.csv')
train_intensity_2_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
train_pitch_2_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_2/train/duration_pitch.csv')
train_pitch_2_syll_table.drop(columns='duration', inplace=True)

dev_formant_2_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_2/dev/duration_formant.csv')
dev_intensity_2_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_2/dev/duration_intensity.csv')
dev_intensity_2_syll_table.drop(columns=['duration', 'intensity1', 'intensity10'], inplace=True)
dev_pitch_2_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_2/dev/duration_pitch.csv')
dev_pitch_2_syll_table.drop(columns='duration', inplace=True)

train_formant_3_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_3/train/duration_formant.csv')
train_intensity_3_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_3/train/duration_intensity.csv')
train_intensity_3_syll_table.drop(columns=['duration', 'intensity1', 'intensity10'], inplace=True)
train_pitch_3_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_3/train/duration_pitch.csv')
train_pitch_3_syll_table.drop(columns='duration', inplace=True)

dev_formant_3_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_3/dev/duration_formant.csv')
dev_intensity_3_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_3/dev/duration_intensity.csv')
dev_intensity_3_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
dev_pitch_3_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_3/dev/duration_pitch.csv')
dev_pitch_3_syll_table.drop(columns='duration', inplace=True)

train_formant_4_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_4/train/duration_formant.csv')
train_intensity_4_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_4/train/duration_intensity.csv')
train_intensity_4_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
train_pitch_4_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_4/train/duration_pitch.csv')
train_pitch_4_syll_table.drop(columns='duration', inplace=True)

dev_formant_4_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_4/dev/duration_formant.csv')
dev_intensity_4_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_4/dev/duration_intensity.csv')
dev_intensity_4_syll_table.drop(columns=['duration','intensity1', 'intensity10'], inplace=True)
dev_pitch_4_syll_table = pd.read_csv('/Users/zelalem/Documents/praat/Data/syllable_4/dev/duration_pitch.csv')
dev_pitch_4_syll_table.drop(columns='duration', inplace=True)


def clean_tables(df_dict):
    '''
    drops all the nans from table, sorts them first by their file name index, then by their syllable location
    '''
    key_list = ['one syllable', 'two syllable', 'three syllable', 'four syllable']
    key_list2 = ['train', 'dev']
    for key in key_list:
        for key2 in key_list2:
            df_list = df_dict[key][key2]
            new_list = []
            for dataframe in df_list:
                dataframe.dropna(inplace=True)
                dataframe_key = [extract_number(string) for string in dataframe['fileName']]
                dataframe['key'] = dataframe_key
                dataframe.sort_values(by=['key', 'name'], inplace=True)
                new_index = [i for i in range(len(dataframe))]
                dataframe.set_index(pd.Index(new_index), inplace=True)
                new_list.append(dataframe.drop(columns=['key', 'fileName', 'name']))
            df_dict[key][key2] = new_list
    return df_dict

def add_data(df_dict):
    '''
    adds a mean and max of pitch and intensity for the tables
    '''
    key_list = ['one syllable', 'two syllable', 'three syllable', 'four syllable']
    key_list2 = ['train', 'dev']
    for key in key_list:
        for key2 in key_list2:
            intensity_df = df_dict[key][key2][1]
            pitch_df = df_dict[key][key2][-1]

            intensity_df = intensity_df.astype('float32')
            max_intensity = [max(tens[0], tens[1], tens[2], tens[3], tens[4], tens[5], tens[6], tens[7]) for 
                    tens in zip(intensity_df['intensity2'], intensity_df['intensity3'], intensity_df['intensity4'],
                                intensity_df['intensity5'], intensity_df['intensity6'], intensity_df['intensity7'], intensity_df['intensity8'],
                                intensity_df['intensity9'])]
            intensity_df['max_intensity'] = max_intensity
            df_dict[key][key2][1] = intensity_df

            pitch_df = pitch_df.astype('float32')
            mean_pitch = [torch.mean(torch.Tensor([tens[0], tens[1], tens[2], tens[3], tens[4], tens[5], tens[6], tens[7], tens[8], tens[9]])).item() for 
                    tens in zip(pitch_df['Pitch1'], pitch_df['Pitch2'], pitch_df['Pitch3'], pitch_df['Pitch4'],
                             pitch_df['Pitch5'], pitch_df['Pitch6'], pitch_df['Pitch7'], pitch_df['Pitch8'],
                             pitch_df['Pitch9'], pitch_df['Pitch10'])]
            pitch_df['mean_pitch'] = mean_pitch
            df_dict[key][key2][-1] = pitch_df
    return df_dict

def consolidate_tables(df_dict):
    '''
    builds the train and dev tables for each syllable type
    '''
    key_list = ['train', 'dev']
    key_list2 = ['one syllable', 'two syllable', 'three syllable', 'four syllable']
    source = ['/Users/zelalem/Documents/LIN_487/one_syllable_data/', '/Users/zelalem/Documents/LIN_487/Jordanian_data_2_syllables/',
              '/Users/zelalem/Documents/LIN_487/Jordanian_data_3_syllables/', '/Users/zelalem/Documents/LIN_487/four_syllable_data/']
    name_list = ['one_syllable_data_', 'Jordanian_dataset_2_syllable_', 'Jordanian_dataset_3_syllable_', 'four_syllable_data_']

    data_list = []
    for key in key_list:
        current_list = []
        for i, key2 in enumerate(key_list2):
            df_list = df_dict[key2][key]
            target_df = pd.read_csv(f'{source[i]}{name_list[i]}{key}.csv')
            target = [i for i in target_df['stress']]
            df = pd.concat(df_list, axis=1)
            reindex(df)
            binarized = binarize(target, i + 1)
            df['Y'] = binarized
            current_list.append(df)
        consolidated = pd.concat(current_list)
        reindex(consolidated)
        data_list.append(consolidated)

    return data_list[0], data_list[-1]

    
def binarize(lst, n):
    '''
    turns the input list into an n binary list
    '''
    bits = [torch.zeros(n) for i in lst]
    target = torch.Tensor(lst) - 1
    for i, bit in enumerate(bits):
        bit[int(target[i].item())] = 1
    result = [int(i)for bit in bits for i in bit]
    return result

def reindex(dataframe):
    '''
    properly indexes this dataframe to account for pandas weird index handling mechanism
    '''
    new_index = [i for i in range(len(dataframe))]
    dataframe.set_index(pd.Index(new_index), inplace=True)


def normalize_tables(df_dict):
    '''
    z normalizes the tables and turns them into a format for a tensor
    '''
    train_table, dev_table = consolidate_tables(df_dict)

    train_table = train_table.sample(frac=1)
    torch.save(torch.Tensor(train_table.index.to_numpy()), ('/Users/zelalem/Documents/praat/Data/train/train_indices.pt'))
    dev_table = dev_table.sample(frac=1)
    torch.save(torch.Tensor(dev_table.index.to_numpy()), ('/Users/zelalem/Documents/praat/Data/dev/dev_indices.pt'))

    print(train_table.head(10))
    print(dev_table.head(10))
    train_y = train_table['Y']
    dev_y = dev_table['Y']

    train_table.drop(columns='Y', inplace=True)
    dev_table.drop(columns='Y', inplace=True)

    print(train_table.dtypes)
    if False:
        train_data = train_table
        dev_data = dev_table
    if True:
        train_data = train_table.astype('float32')
        dev_data = dev_table.astype('float32')
        train_data = (train_data - train_data.mean()) / train_data.std()
        dev_data = (dev_data - dev_data.mean()) / dev_data.std()
    if False:
        train_marker = [0 for i in range(len(train_table))]
        train_table['marker'] = train_marker
        dev_marker = [1 for i in range(len(dev_table))]
        dev_table['marker'] = dev_marker
        new_table = pd.concat([train_table, dev_table])
        reindex(new_table)
        train_indexes = new_table.index[new_table['marker'] == 0]
        dev_indexes = new_table.index[new_table['marker'] == 1]
        data = new_table - new_table.mean() / new_table.std()
        train_data = data.loc[train_indexes]
        dev_data = data.loc[dev_indexes]
        train_indexes = list(train_indexes).sort()
        dev_indexes = list(dev_indexes).sort()
        train_data['key'] = train_indexes
        dev_data['key'] = dev_indexes
        train_data.sort_values(by='key', inplace=True)
        dev_data.sort_values(by='key', inplace=True)
        train_data.drop(columns='key', inplace=True)
        dev_data.drop(columns='key', inplace=True)
        train_data.drop(columns='marker', inplace=True)
        dev_data.drop(columns='marker', inplace=True)

    train_data = torch.Tensor(train_data.to_numpy()), torch.tensor(train_y.to_numpy()).long()
    dev_data = torch.Tensor(dev_data.to_numpy()), torch.tensor(dev_y.to_numpy()).long()

    print('train shape: ', train_data[0].shape, 
          '1 syllables: ', len(df_dict['one syllable']['train'][0]),
          ', 2 syllables: ', len(df_dict['two syllable']['train'][0]), 
          ', 3 syllables: ', len(df_dict['three syllable']['train'][0]), 
          ', 4 syllables :', len(df_dict['four syllable']['train'][0]),
          train_data[1].shape)
    print('dev shape: ', dev_data[0].shape,
          '1 syllables: ', len(df_dict['one syllable']['dev'][0]),
          ', 2 syllables: ', len(df_dict['two syllable']['dev'][0]), 
          ', 3 syllables: ', len(df_dict['three syllable']['dev'][0]),
          ', 4 syllables: ', len(df_dict['four syllable']['dev'][0]),
          dev_data[1].shape)

    torch.save(train_data[0],'/Users/zelalem/Documents/praat/Data/train/train_input.pt')
    torch.save(train_data[1],'/Users/zelalem/Documents/praat/Data/train/train_label.pt')

    torch.save(dev_data[0],'/Users/zelalem/Documents/praat/Data/dev/dev_input.pt')
    torch.save(dev_data[1],'/Users/zelalem/Documents/praat/Data/dev/dev_label.pt')

def extract_number(str):
    '''
    helper function extracts a number from the filename
    '''
    return int("".join(re.findall(r'\d*', str)))



def generate_dataset(df_dict):
    '''
    automater function
    '''
    cleaned_df = clean_tables(df_dict)
    added_df = add_data(cleaned_df)
    normalize_tables(added_df)

df_dict = {'one syllable':{'train':[train_formant_1_syll_table, train_intensity_1_syll_table, train_pitch_1_syll_table], 
           'dev':[dev_formant_1_syll_table, dev_intensity_1_syll_table, dev_pitch_1_syll_table]},
           'two syllable':{'train':[train_formant_2_syll_table, train_intensity_2_syll_table, train_pitch_2_syll_table],
            'dev':[dev_formant_2_syll_table, dev_intensity_2_syll_table, dev_pitch_2_syll_table]},
            'three syllable':{'train':[train_formant_3_syll_table, train_intensity_3_syll_table, train_pitch_3_syll_table],
            'dev':[dev_formant_3_syll_table, dev_intensity_3_syll_table, dev_pitch_3_syll_table]},
            'four syllable':{'train':[train_formant_4_syll_table, train_intensity_4_syll_table, train_pitch_4_syll_table],
            'dev':[dev_formant_4_syll_table, dev_intensity_4_syll_table, dev_pitch_4_syll_table]}}

generate_dataset(df_dict)