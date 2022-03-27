''' 
    Author: Vu Quoc Hien
    Date created: 2022-03-14
'''

from tkinter import Variable
import numpy as np
import torch as T
from src.data_helper import *

from models.SharedNet2C import SharedNet2C
from models.SharedNet1C import SharedNet1C

def transform_dataset(dataset):
        """
        Transform dataset to batchs
        :param dataset: dataset
        :param mode: train or test
        :return: transformed dataset
        """
        inputs = []
        targets = []
        device = 'cuda' if T.cuda.is_available() else 'cpu'
        for i in range(0, len(dataset['data']), 2):
            row = dataset['data'][i]
            row_1 = [row[0], row[0]]
            row_2 = [row[1], row[1]]
            row = [row_1, row_2]
            row = T.FloatTensor(np.array(row).astype(np.float64)).to(device)
            labels = [dataset['label'][i], dataset['label'][i]]
            labels = T.FloatTensor(np.array(labels).astype(np.float64)).to(device)
            
            inputs.append(row)
            targets.append(labels)
            
        return (inputs, targets)

def collection(dataset_path=None,
                dataset=None,
                model=None,
                model_path=None):
    if dataset is None:
        ''' load csv file '''
        dataset = load_from_bin(dataset_path)
        # add feature path
    if model is None:
        if dataset['type'] == "2C":
            model = SharedNet2C(dataset['input-size'], dataset['outputs-size'])
        elif dataset['type'] == "1C":
            model = SharedNet1C(dataset['input-size'], dataset['outputs-size'])
        else:
            raise ValueError("Dataset type is not supported")
        if model_path is not None:
            model.load(path=model_path)
            
    # model._eval()
    
    new_dataset = {
        'data': [],
        'label': [],
        'columns': dataset['columns'],
        'outputs-size': dataset['outputs-size'],
        'input-size': dataset['input-size'][1],
        'type': '1C',
        'path': dataset['path'].replace('2C', 'co-1C')
    }
    
    n_y_true = 0  
    with T.no_grad():       # Tắt gradient các tensor trong khối lệnh phía dưới 
        dataset = transform_dataset(dataset)
        for idx, (inputs, targets) in enumerate(zip(dataset[0], dataset[1])):
            X1_outputs = model(inputs[0][0], inputs[0][1]).flatten().cpu().numpy()
            X2_outputs = model(inputs[1][0], inputs[1][1]).flatten().cpu().numpy()
            presize = 0
            allele_targets = []
            is_candidate = True
            for name, output_size in model.outputs_size:
                allele_out_0 = X1_outputs[presize:presize + output_size].argsort()[-1]
                allele_out_1 = X2_outputs[presize:presize + output_size].argsort()[-1]
                allele_targets_index = targets[0][presize:presize + output_size]
                allele_targets = allele_targets_index.argsort().cpu().numpy()[-2:][::-1]
                
                if allele_targets_index[allele_targets[1]] == 0:
                    allele_targets[1] = allele_targets[0]
                    
                if allele_out_0 not in allele_targets or \
                        allele_out_1 not in allele_targets:
                    is_candidate = False
                    presize += output_size
                    break
                if allele_targets[0] != allele_targets[1] and \
                    allele_out_0 == allele_out_1:
                    is_candidate = False
                    presize += output_size
                    break
                if allele_targets[0] != allele_targets[1]:
                    if allele_out_0 == allele_targets[0]:
                        targets[0][presize + allele_targets[1]] = 0
                        targets[1][presize + allele_targets[0]] = 0
                    else:
                        targets[0][presize + allele_targets[0]] = 0
                        targets[1][presize + allele_targets[1]] = 0
                presize += output_size
            
            if is_candidate:
                data = inputs.cpu().numpy()
                label = targets.cpu().numpy()
                new_dataset['data'].append(data[0][0])
                new_dataset['label'].append(label[0])
                new_dataset['data'].append(data[1][0])
                new_dataset['label'].append(label[1])
                
    save_to_bin(new_dataset, new_dataset['path'])
    print("Number of collection data: {}, ({}%)".format(len(new_dataset['data']),
                                                        len(new_dataset['data'])/len(dataset[0])*100))
    return new_dataset 
            
            
            