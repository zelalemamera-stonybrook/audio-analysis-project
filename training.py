import pandas as pd
import re
import torch

formant_table = pd.read_csv('/Users/zelalem/Documents/MFA/praat/Data/dev/duration_formant.csv')
intensity_table = pd.read_csv('/Users/zelalem/Documents/MFA/praat/Data/dev/duration_intensity.csv')
pitch_table = pd.read_csv('/Users/zelalem/Documents/MFA/praat/Data/dev/duration_pitch.csv')
train_table = pd.read_csv('/Users/zelalem/Documents/LIN_487/Jordanian_data_3_syllables/Jordanian_dataset_3_syllable_dev.csv')

formant_table.dropna(inplace=True)

def extract_number(str):
    return int("".join(re.findall(r'\d*', str)))

formant_key = [extract_number(string) for string in formant_table['fileName']]
intensity_key = [extract_number(string) for string in intensity_table['fileName']]
pitch_key = [extract_number(string) for string in pitch_table['fileName']]

formant_table['key'] = formant_key
intensity_table['key'] = intensity_key
pitch_table['key'] = pitch_key

formant_table.sort_values(by=['key', 'name'], inplace=True)
intensity_table.sort_values(by=['key', 'name'], inplace=True)
pitch_table.sort_values(by=['key', 'name'], inplace=True)

new_index = [i for i in range(len(formant_table))]
new_index2 = [i for i in range(len(intensity_table))]
new_index3 = [i for i in range(len(pitch_table))]

formant_table.set_index(pd.Index(new_index), inplace=True)
intensity_table.set_index(pd.Index(new_index2), inplace=True)
pitch_table.set_index(pd.Index(new_index3), inplace=True)

formant_table.drop(columns='key', inplace=True)
intensity_table.drop(columns='key', inplace=True)
pitch_table.drop(columns='key', inplace=True)

max_intensity = [max(tens[0], tens[1], tens[2], tens[3], tens[4], tens[5], tens[6], tens[7], tens[8], tens[9]) for 
                 tens in zip(intensity_table['intensity1'],intensity_table['intensity2'],intensity_table['intensity3'],intensity_table['intensity4'],
                             intensity_table['intensity5'],intensity_table['intensity6'],intensity_table['intensity7'],intensity_table['intensity8'],
                             intensity_table['intensity9'],intensity_table['intensity10'])]
intensity_table['maxIntensity'] = max_intensity

mean_pitch = [torch.mean(torch.Tensor([tens[0], tens[1], tens[2], tens[3], tens[4], tens[5], tens[6], tens[7], tens[8], tens[9]])).item() for 
                 tens in zip(pitch_table['Pitch1'],pitch_table['Pitch2'],pitch_table['Pitch3'],pitch_table['Pitch4'],
                             pitch_table['Pitch5'],pitch_table['Pitch6'],pitch_table['Pitch7'],pitch_table['Pitch8'],
                             pitch_table['Pitch9'], pitch_table['Pitch10'])]
pitch_table['mean_pitch'] = mean_pitch

max_pitch = [max(tens[0], tens[1], tens[2], tens[3], tens[4], tens[5], tens[6], tens[7], tens[8], tens[9]) for 
                 tens in zip(pitch_table['Pitch1'],pitch_table['Pitch2'],pitch_table['Pitch3'],pitch_table['Pitch4'],
                             pitch_table['Pitch5'],pitch_table['Pitch6'],pitch_table['Pitch7'],pitch_table['Pitch8'],
                             pitch_table['Pitch9'], pitch_table['Pitch10'])]
pitch_table['max_pitch'] = max_pitch

if True:
    print("formant table: ", formant_table.head(15))
    print("intensity table: ", intensity_table.head(15))
    print("pitch table: ", pitch_table.head(15))
if False:
    intensity_data = intensity_table.drop(columns=['fileName', 'name', 'duration'])
    pitch_data = pitch_table.drop(columns=['fileName', 'name', 'duration'])
    full_table = pd.concat([formant_table, intensity_data, pitch_data], axis=1, ignore_index=False)
    print(full_table.head(15))
    start = full_table[full_table['fileName'] == 'file0.TextGrid']
    start.sort_values(by='name', inplace=True)
    start.drop(columns=['fileName', 'name'], inplace=True)
    start_data = (start - start.mean())
    for i in range(len(train_table)):
        if i == 0: continue
        name = f"file{i}.TextGrid"
        current = full_table[full_table['fileName'] == name]
        current.sort_values(by='name', inplace=True)
        current.drop(columns=['fileName', 'name'], inplace=True)
        current_data = (current - current.mean())
        start_data = pd.concat([start_data, current_data])
if True:
    intensity_data = intensity_table.drop(columns=['fileName', 'name', 'duration'])
    pitch_data = pitch_table.drop(columns=['fileName', 'name', 'duration'])
    full_table = pd.concat([formant_table, intensity_data, pitch_data], axis=1, ignore_index=False)
    if True:
        full_table.drop(columns='duration', inplace=True)
        full_table.drop(columns='meanIntensity', inplace=True)
        full_table.drop(columns='maxIntensity', inplace=True)
        full_table.drop(columns='intensity1', inplace=True)
        full_table.drop(columns='intensity6', inplace=True)
        full_table.drop(columns='intensity9', inplace=True)
        full_table.drop(columns='intensity10', inplace=True)
        full_table.drop(columns='Pitch4', inplace=True)
        full_table.drop(columns='Pitch7', inplace=True)
        full_table.drop(columns='Pitch8', inplace=True)
        full_table.drop(columns='Pitch2', inplace=True)
        full_table.drop(columns='Pitch3', inplace=True)
        full_table.drop(columns='Pitch10', inplace=True)
        full_table.drop(columns='Pitch9', inplace=True)
        full_table.drop(columns='Pitch6', inplace=True)
        full_table.drop(columns='Pitch1', inplace=True)
        full_table.drop(columns='Pitch5', inplace=True)
        full_table.drop(columns='mean_pitch', inplace=True)
        full_table.drop(columns='max_pitch', inplace=True)
        full_table.drop(columns='F1', inplace=True)
        full_table.drop(columns='F2', inplace=True)
        full_table.drop(columns='F3', inplace=True)
    print(full_table.head(15))
    start = full_table.drop(columns=['fileName', 'name'])
    start_data = (start - start.mean()) / start.std()

labels = [value - 1 for value in train_table['stress']]
tuples = [[0,0,0] for value in train_table['stress']]
for i in range(len(labels)):
    tuples[i][labels[i]] = 1
target = [i for tup in tuples for i in tup]

#start_data['Y'] = target
#ones = start_data[start_data['Y'] == 1]
#start_data = pd.concat([start_data, ones])
#start_data = start_data.sample(frac=1)
#target = start_data['Y']

Y = torch.Tensor(target)

training_data = torch.Tensor(start_data.to_numpy())


print(training_data.shape)
print(Y.shape)

Y = Y.long()
x = (Y.sum() / len(Y)) * 100
print(x)
torch.save(training_data, '/Users/zelalem/Documents/MFA/praat/Data/dev/tensors/dev_input.pt')
torch.save(Y, '/Users/zelalem/Documents/MFA/praat/Data/dev/tensors/dev_label.pt')