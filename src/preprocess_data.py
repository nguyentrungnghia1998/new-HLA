from src.data_helper import *

def pretrain(data_path=None, index_path=None, label_path=None, 
             ref_pos_path=None, pos_path=None,
             hla_types=None, nrows=None, saved=True):
    
    ''' read vcf file and convert to csv format '''
    df = load_vcf_file(data_path, index_path=index_path,ref_pos_path=ref_pos_path,
                       nrows=nrows, saved=False)
    label_df_file_path = label_path
    label_df = pd.read_csv(label_df_file_path, index_col=0)
    label_df = label_df.set_index(np.array(['_'.join(x.split('_')[:4]) 
                                            for x in label_df.index.to_numpy()]))
    label_df.sort_index(inplace=True)
    label_df = label_df.fillna('nan') # fill nan in label_df with '0'
    
    one_hot_encoder = {}
    columns = []
    for hla in hla_types:
        columns.extend([hla.upper() + '_' + x for x in ['1', '2']])
    
    ''' make onehot encoder with two columns for each hla type '''
    for i in range(0, len(columns), 2):
        col_1 = label_df[columns[i]].values
        col_2 = label_df[columns[i+1]].values
        combined_col = sorted(set(np.concatenate((col_1, col_2), axis=0)))
        n_values = len(combined_col) + 1
        one_hot_vector = np.eye(n_values)[range(n_values)]
        for j, x in enumerate(combined_col):
            one_hot_encoder[(columns[i], x)] = one_hot_vector[j]
            one_hot_encoder[(columns[i+1], x)] = one_hot_vector[j]
        
    label_df_one_hot = label_df.copy()
    for col in columns:
        label_df_one_hot[col] = label_df_one_hot[col].apply(lambda x: one_hot_encoder[(col, x)])
    
    ''' One hot encoding '''
    allele_labels_1 = columns[::2]
    allele_labels_2 = columns[1::2]
    df_allele_labels_1 = label_df_one_hot[allele_labels_1]
    df_allele_labels_2 = label_df_one_hot[allele_labels_2]
    df_allele_labels_1.rename(index=lambda x: x + '_1', inplace=True)
    df_allele_labels_1.rename(columns=lambda x: x.replace('_1', ''), inplace=True)
    df_allele_labels_2.rename(index=lambda x: x + '_2', inplace=True)
    df_allele_labels_2.rename(columns=lambda x: x.replace('_2', ''), inplace=True)
    df_allele_labels = pd.concat([df_allele_labels_1, df_allele_labels_2], axis=0)
    columns = df_allele_labels.columns
    
    dropped_label_indices = []
    dropped_data_indices = []
    
    ''' Drop rows in df_allele_labels that not in df '''
    for i in range(len(df_allele_labels)):
        if df_allele_labels.index[i] not in df.index:
            dropped_label_indices.append(i)

    ''' Drop rows in df that not in df_allele_labels '''
    for i in range(len(df)):
        if df.index[i] not in df_allele_labels.index:
            dropped_data_indices.append(i)

    df_allele_labels.drop(df_allele_labels.index[dropped_label_indices], inplace=True)
    df.drop(df.index[dropped_data_indices], inplace=True)
    
    ''' concat of all columns is label_df_one_hot '''
    df_allele_labels['label'] = df_allele_labels.apply(
        lambda x: np.concatenate(x.values), axis=1)
    
    dataset = {
        'data': [],
        'label': []
    }
    
    dataset_length = len(df) // 2
    
    for i in range(dataset_length):
        data_row = np.concatenate(([df.iloc[i].values], [df.iloc[i + dataset_length].values]))
        label = np.logical_or(df_allele_labels['label'].iloc[i], 
                                df_allele_labels['label'].iloc[i + dataset_length]) * 1
        dataset['data'].append(data_row)
        dataset['label'].append(label)
        # swap two rows 0, 1
        data_row = np.array([data_row[1], data_row[0]])
        dataset['data'].append(data_row)
        dataset['label'].append(label)
        
    dataset['input-size'] = (2, len(dataset['data'][0][0]))
    dataset['type'] = "2C"   
        
    dataset['columns'] = columns
    dataset['outputs-size'] = [['HLA_' + col, len(df_allele_labels[col].iloc[0])] for col in columns]
    dataset['path'] = data_path.replace('.vcf.gz', '.' + dataset['type'] + '.' + \
            '_'.join(hla_types) + '.bin')
    
    if saved:
        save_to_bin(dataset, dataset['path'])
    
    return dataset
    
