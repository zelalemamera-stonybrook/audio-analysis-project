import torch.nn as nn
import torch
import sklearn.cluster as skl
import pandas as pd
import numpy as np

class LogisticRegressor(nn.Module):
    '''
    implements a logistic regressor over the input dataset. 
    '''
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.weights = nn.Linear(input_dimension, output_dimension)
        self.shape = (input_dimension, output_dimension)
        self.name = 'logistic regressor'

    def forward(self, input_data):
        softmax = nn.Softmax(dim=1)
        output = softmax(self.weights(input_data))
        return output

class FeedforwardNeuralNetwork(nn.Module):
    '''
    implements several linear layers with activation functions
    '''
    def __init__(self, shape):
        '''
        shape must be a tuple specifying input and output dimensions in addition to any hidden layers.
        '''
        super().__init__()
        parameters = torch.Tensor([shape[i] * shape[i + 1] + shape[i + 1] for i in range(len(shape) - 1)])
        print('model parameters:', parameters.sum().item(), '\nsize on disk:', ((parameters.sum() * 32) / 10**6).item(), "MB")
        weights = [nn.Linear(shape[i], shape[i+1]) for i in range(len(shape) - 1)]
        self.num_of_layers= len(weights)
        self.shape = shape
        self.name = 'neural network'
        for i, weight in enumerate(weights):
            setattr(self, f"layer{i}", weight)
    
    def forward(self, input):
        softmax = nn.Softmax(dim=1)
        current = input
        activate = nn.ReLU()
        for i in range(self.num_of_layers - 1):
            current = activate(getattr(self, f"layer{i}").forward(current))
        output = softmax(getattr(self, f"layer{self.num_of_layers - 1}").forward(current))
        return output
    
class KMeans(skl.KMeans):
    '''
    uses a clustering model from sklearn
    '''
    def __init__(self, k):
        super().__init__(n_clusters=k)
        self.k = k
        

def train(model, input, target, goal, max):
    '''
    updates the weights of the model to match its output closer to the distribution provided by target
    '''
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    d = 0
    n = 0
    last_improvement = 0
    while(d < goal and n < max and last_improvement < 1000):
        model.train()
        optimizer.zero_grad()
        output = model.forward(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        accuracy_new = accuracy(model, input, target, False, [])
        if accuracy_new >= d:
            last_improvement +=1
        else:
            last_improvement = 0
        d = accuracy_new
        n += 1
    return d, n

def context_assumption(model_predictions, indices, word):
    '''
    adjusts the model predictions based on the linguistic assumption that words should have only one
    primary stress and one.
    '''
    df = pd.DataFrame({'predictions':list(model_predictions), 'index':indices}, index=range(len(model_predictions)))
    df.sort_values(by='index', inplace=True)
    n1 = 35
    n2 = 284
    n3 = 153
    n4 = 20

    dev_1_table = df[[i < n1 for i in df['index']]]
    dev_1_predictions = [i.tolist() for i in dev_1_table['predictions']]

    dev_2_table = df[[ n1 <= i and i < n1 + n2 for i in df['index']]]
    dev_2_predictions = [i for i in dev_2_table['predictions']]

    dev_3_table = df[[n1 + n2 <= i and i < n1 + n2 + n3 for i in df['index']]]
    dev_3_predictions = [i for i in dev_3_table['predictions']]

    dev_4_table = df[[n1 + n2 + n3 <= i for i in df['index']]]
    dev_4_predictions = [i for i in dev_4_table['predictions']]

    output_list = []
    dev1 = torch.Tensor(dev_1_predictions)
    output_list += torch.argmax(dev1,dim=1).tolist()

    if True:
        for i in range(int(n2 / 2)):
            stress1 = dev_2_predictions[i * 2][1].item()
            stress2 = dev_2_predictions[i * 2 + 1][1].item()
            probs = torch.Tensor([stress1, stress2])
            new_list = [0,0]
            new_list[torch.argmax(probs).item()] = 1
            output_list += new_list
    if(len(dev_3_predictions) != 0):
        for i in range(int(n3/3)):
            stress1 = dev_3_predictions[i * 3][1].item()
            stress2 = dev_3_predictions[i * 3 + 1][1].item()
            stress3 = dev_3_predictions[i * 3 + 2][1].item()
            probs = torch.Tensor([stress1, stress2, stress3])
            new_list = [0,0,0]
            new_list[torch.argmax(probs).item()] = 1
            output_list += new_list
    for i in range(int(n4 / 4)):
        stress1 = dev_4_predictions[i * 4][1].item()
        stress2 = dev_4_predictions[i * 4 + 1][1].item()
        stress3 = dev_4_predictions[i * 4 + 2][1].item()
        stress4 = dev_4_predictions[i * 4 + 2][1].item()
        probs = torch.Tensor([stress1, stress2, stress3, stress4])
        new_list = [0,0,0,0]
        new_list[torch.argmax(probs).item()] = 1
        output_list += new_list
    if word:
        output = []
        ones = output_list[:n1]
        for i, val in enumerate(ones):
            if val == 0:
                ones[i] = 5
        output += torch.ones(35).tolist()
        twos = turn_to_words(output_list[n1:n1 + n2], 2)
        output += twos
        threes = turn_to_words(output_list[n1 + n2: n1 + n2 + n3], 3)
        output += threes
        fours = turn_to_words(output_list[n1 + n2 + n3:], 4)
        output += fours
        return output
    df['adjusted'] = output_list
    df.sort_index(inplace=True)
    correct = [i for i in df['adjusted']]
    return torch.Tensor(correct)

def turn_to_words(lst, n):
    '''
    assumes this is a list of n grams and turns it into the presumed integer from the class
    '''
    output = []
    for i in range(int(len(lst) / n)):
        n_gram = torch.Tensor(lst[ i * n : i * n + n])
        classed = torch.argmax(n_gram).item() + 1
        output.append(classed)
    return output

def accuracy(model, input, target, context, indices):
   '''
   computes the accuracy of this model against the target
   '''
   model.eval()
   if context:
       output = context_assumption(model.forward(input), indices)
   else:
       output = torch.argmax(model.forward(input), dim=1)
   accuracy = (1 - ((torch.sum(torch.abs(output - target )) / len(target))))
   return accuracy

def precision(model, input, target, context, indices):
    '''
    computes the precision as the proportion of true positives to all positives
    '''
    if context:
       output = context_assumption(model.forward(input), indices)
    else:
       output = torch.argmax(model.forward(input), dim=1)
    total_positives_claimed = output.sum()
    true_positives = 0
    for i in range(len(output)):
        if output[i] == target[i] and target[i] == 1:
            true_positives+=1
    return true_positives / total_positives_claimed 

def recall(model, input, target, context, indices):
    '''
    compites the recall of this model as the proportion of correctly identified positives.
    '''
    if context:
       output = context_assumption(model.forward(input), indices)
    else:
       output = torch.argmax(model.forward(input), dim=1)
    total_positives = target.sum()
    true_positives = 0
    for i in range(len(output)):
        if output[i] == target[i] and target[i] == 1:
            true_positives+=1
    return true_positives / total_positives

def fscore(model, input, target, context, indices):
    '''
    computes the f score of this model using precision and recall
    '''
    p = precision(model, input, target, context, indices)
    r = recall(model, input, target, context, indices)
    return 2 * (p * r) / (p + r)

def word_accuracy(model, input, target, context, indices):
    '''
    computes the accuracy of this model on the set of words as opposed to syllables.
    '''
    model.eval()
    if context:
        output = context_assumption(model.forward(input), indices, True)
        confusion_matrix = get_confusion_matrix(output, target)
        counts = 0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                if i == j:
                    counts += confusion_matrix[i][j]
        return counts / len(target)
    
def word_precision(model, input, target, context, indices):
    '''
    '''
    model.eval()
    if context:
        output = context_assumption(model.forward(input), indices, True)
        confusion_matrix = get_confusion_matrix(output, target)
        precision_list = []
        n = len(confusion_matrix)
        confusion_matrix = confusion_matrix.T
        for i, row in enumerate(confusion_matrix):
            if i < n:
                claims = row.sum().item()
                true = row[i].item()
                precision_list.append(true/claims)
            else:
                continue
        return precision_list

def word_recall(model, input, target, context, indices):
    '''
    '''
    model.eval()
    if context:
        output = context_assumption(model.forward(input), indices, True)
        confusion_matrix = get_confusion_matrix(output, target)
        recall_list = []
        for i, row in enumerate(confusion_matrix):
            total = row.sum().item()
            true = row[i].item()
            recall_list.append( true / total)
        return recall_list

def word_fscore(model, input, target, context, indices):
    '''
    '''
    model.eval()
    p = word_precision(model, input, target, context, indices)
    r = word_recall(model, input, target, context, indices)
    p = torch.Tensor(p).mean().item()
    r = torch.Tensor(r).mean().item()
    return 2 * (p * r) / (p + r)

def get_confusion_matrix(claimed, target):
    '''
    gives the multiclass analysis between these two vectors.
    '''
    columns = set(claimed)
    rows = set(target)

    matrix = torch.zeros((len(rows), len(columns)))
    for i in range(len(rows)):
        for j in range(len(columns)):
            matrix[i][j] = count_claim(j + 1, i + 1, claimed, target)
    print(matrix)
    return matrix

def count_claim(j, i, claimed, target):
    '''
    the model makes the claim j for the true class i, count how many of them and return the result
    '''
    claimed_total = 0
    for model, true in zip(claimed, target):
        if true == i:
            if model == j:
                claimed_total+=1
    return claimed_total

train_input = torch.load('/Users/zelalem/Documents/praat/Data/train/train_input.pt')
train_label = torch.load('/Users/zelalem/Documents/praat/Data/train/train_label.pt')
dev_input = torch.load('/Users/zelalem/Documents/praat/Data/dev/dev_input.pt')
dev_label = torch.load('/Users/zelalem/Documents/praat/Data/dev/dev_label.pt')

train_indices = torch.load('/Users/zelalem/Documents/praat/Data/train/train_indices.pt')
dev_indices = torch.load('/Users/zelalem/Documents/praat/Data/dev/dev_indices.pt')

train_1_table = pd.read_csv('/Users/zelalem/Documents/LIN_487/one_syllable_data/one_syllable_data_train.csv')
train_2_table = pd.read_csv('/Users/zelalem/Documents/LIN_487/Jordanian_data_2_syllables/Jordanian_dataset_2_syllable_train.csv')
train_3_table = pd.read_csv('/Users/zelalem/Documents/LIN_487/Jordanian_data_3_syllables/Jordanian_dataset_3_syllable_train.csv')
train_4_table = pd.read_csv('/Users/zelalem/Documents/LIN_487/four_syllable_data/four_syllable_data_train.csv')

dev_1_table = pd.read_csv('/Users/zelalem/Documents/LIN_487/one_syllable_data/one_syllable_data_dev.csv')
dev_2_table = pd.read_csv('/Users/zelalem/Documents/LIN_487/Jordanian_data_2_syllables/Jordanian_dataset_2_syllable_dev.csv')
dev_3_table = pd.read_csv('/Users/zelalem/Documents/LIN_487/Jordanian_data_3_syllables/Jordanian_dataset_3_syllable_dev.csv')
dev_4_table = pd.read_csv('/Users/zelalem/Documents/LIN_487/four_syllable_data/four_syllable_data_dev.csv')
dev_table = pd.concat([dev_1_table, dev_2_table, dev_3_table, dev_4_table])


if True:
    print(train_input.shape, train_label.shape)
    print(dev_input.shape, dev_label.shape)

def test(model, test_input, test_target, context, indices, word):
    '''
    runs a bunch of diagnostics on the model to see how well it is able to generalize to unseen data
    '''
    if word:
        a = word_accuracy(model, test_input, test_target, context, indices)
        p = word_precision(model, test_input, test_target, context, indices)
        r = word_recall(model, test_input, test_target, context, indices)
        f = word_fscore(model, test_input, test_target, context, indices)
    else:
        a = accuracy(model, test_input, test_target, context, indices)
        p = precision(model, test_input, test_target, context, indices)
        r = recall(model, test_input, test_target, context, indices)
        f = fscore(model, test_input, test_target, context, indices)
    return a, p, r, f

def optimize(model, train_input, train_label, dev_input, dev_label, context, indices, word, dev_table):
    '''
    tries to optimize this model to fit as well as possible to the dev_data 
    '''
    epochs = 0
    fitting = 0
    path = '/Users/zelalem/Documents/model_recordbook/model_logbook.txt'
    with open(path, 'a') as f:
        #f.write("name\tshape\tepochs\tcycles\taccuracy\tprecision\trecall\tfscore\tsetup\n")
        line = ''
        maximum_f = 0
        if word:
            dev_Y = [int(i) for i in dev_table['stress']]
            current_accuracy, current_precision, current_recall, current_f = test(model, dev_input, dev_Y, context, indices, word)
        else:
            current_accuracy, current_precision, current_recall, current_f = test(model, dev_input, dev_label, context, indices, word)
        while current_accuracy < .95 and epochs < 100 and fitting < .95:
            epochs+=1
            fitting, cycles = train(model, train_input, train_label, .95, 100)
            if word:
                dev_Y = [int(i) for i in dev_table['stress']]
                current_accuracy, current_precision, current_recall, current_f = test(model, dev_input, dev_Y, context, indices, word)
            else:
                current_accuracy, current_precision, current_recall, current_f = test(model, dev_input, dev_label, context, indices, word)
            if current_f > maximum_f:
                maximum_f = current_f
                if word:
                    l = [current_precision, current_recall]
                    for lst in l:
                        for i, val in enumerate(lst):
                            lst[i] = round(val, 2)
                    line = f"{model.name}\t{model.shape}\t{round(epochs, 2)}\t{cycles + 100 * (epochs - 1)}\t{round(current_accuracy.item(), 2)}\t{current_precision}\t{current_recall}\t{round(current_f, 2)}\tHw+\n"
                else:
                    line = f"{model.name}\t{model.shape}\t{round(epochs, 2)}\t{cycles + 100 * (epochs - 1)}\t{round(current_accuracy.item(), 2)}\t{round(current_precision.item(), 2)}\t{round(current_recall.item(), 2)}\t{round(current_f.item(), 2)}\tH\n"
        f.write(line)


def analyze_model_results(results, context, index, table_1_syll, table_2_syll, table_3_syll, table_4_syll):
    '''
    given this model's claims on the current dataset, it gives the values in the table for this dataset
    '''
    syllable_1 = table_1_syll
    syllable_2 = pd.concat([table_2_syll.copy(), table_2_syll])
    syllable_2.sort_index(inplace=True)
    syllable_3 = pd.concat([table_3_syll.copy(), table_3_syll.copy(), table_3_syll])
    syllable_3.sort_index(inplace=True)
    syllable_4 = pd.concat([table_4_syll.copy(), table_4_syll.copy(), table_4_syll.copy(), table_4_syll])
    syllable_4.sort_index(inplace=True)

    if context:
        output = context_assumption(results, index, False)
    else:
        output = torch.argmax(results, dim=1)

    table = pd.concat([syllable_1, syllable_2, syllable_3, syllable_4])

    df = pd.DataFrame({'index':list(index), 'results':list(results), 'predicted': list(output)}, index=range(len(results)))
    df.sort_values(by='index', inplace=True)

    lines = [(a[0], (round(a[1][0].item(), 3), round(a[1][1].item(), 3)), a[2].item(), a[3]) for a in zip(table['ipa'], df['results'], df['predicted'], table['stress'])]
    with open('/Users/zelalem/Documents/model_recordbook/results_analysis.txt', 'w') as f:
        f.write('ipa\tprobabilities\tprediction\tcorrect\n')
        for line in lines:
            f.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\n")
    

if False:
    logistic_regressor = LogisticRegressor(train_input.shape[1], 2)
    optimize(logistic_regressor, train_input, train_label, dev_input, dev_label, True, dev_indices, True, dev_table)
    analyze_model_results(logistic_regressor.forward(dev_input), True, dev_indices, dev_1_table, dev_2_table, dev_3_table, dev_4_table)
if True:
    neural_network = FeedforwardNeuralNetwork((train_input.shape[1],128, 2))
    optimize(neural_network, train_input, train_label, dev_input, dev_label, True, dev_indices, True, dev_table)
    analyze_model_results(neural_network.forward(dev_input), True, dev_indices, dev_1_table, dev_2_table, dev_3_table, dev_4_table)


if False:
    with open('/Users/zelalem/Documents/model_recordbook/weights.txt', 'w') as f:
        parameters = next(iter(logistic_regressor.parameters()))
        parameters = parameters.tolist()
        for lst in parameters:
            for i, num in enumerate(lst):
                lst[i] = round(num, 3)
            f.write(f"{lst}\n")

