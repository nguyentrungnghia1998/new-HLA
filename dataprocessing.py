# read ped file and output a list of unique individuals
# Author: Hien VQ
# Date: 2021-12-27

import psutil
import pandas as pd

file_name = 'input/consensus23.phased.HLA.ped'

individuals = []
data = []
# read line by line
with open(file_name, 'r') as f:
    for line in f:
        line = line.strip()
        # skip empty line
        if line == '':
            continue
        # skip comment line
        if line.startswith('#'):
            continue
        # split line by tab
        line = line.split(' ')
        # extract individual name
        individual = line[1]
        # print(individual)
        # add to list
        if individual not in individuals:
            individuals.append(individual)
        data.append(line[6:])
        # print(len(data))
        if len(data) > 10:
            break
        if psutil.virtual_memory().percent > 90:
            print('Warning: Memory usage is too high!')
            break
        
df = pd.DataFrame(data, index=individuals, columns=range(0, len(data[0])))

print(df.head(5))