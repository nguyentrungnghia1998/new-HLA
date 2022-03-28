import pickle
from cyvcf2 import VCF		# Lib to work with vcf files
import pandas as pd
import numpy as np
import seaborn as sns

def load_vcf_file(path, index_path, ref_pos_path=None,saved=True, nrows=None):
    '''
    path: path to vcf file
    return: csv file
    '''
    
    if not path.endswith('.vcf.gz'):		
        raise ValueError('Input file must be a vcf.gz file')
    
    data = {}
    headers = []
    index_list = []
    n_rows = 0
    with open(index_path, 'r') as f:
        index = f.read().splitlines()		# Đọc file chứa đường dẫn các chỉ số được chọn 
    
    index_list.extend([x + '_1' for x in index])
    index_list.extend([x + '_2' for x in index])		# Thêm danh sách chỉ số trên nhưng có thêm đuôi thứ tự
    
    if nrows is None:
        limit_rows = np.inf		# Nếu không khai báo số hàng giới hạn thì coi là vô cùng
    else:
        limit_rows = nrows
    
    if ref_pos_path is not None:
        with open(ref_pos_path, 'r') as f:
            
            ref_position = [int(x) for x in f.read().splitlines()]		# Đọc file chứa đường dẫn các chỉ số được chọn
            for idx in index_list:
                data[idx] = [0] * len(ref_position)		# Khởi tạo giá trị cho các chỉ số được chọn
    else:
        ref_position = None
        for idx in index_list:
            data[idx] = []
    idx = 0
    
    for variant in VCF(path): # or VCF('some.bcf')		# Đọc các biến dị trong file vcf đã cho 
        if ref_position is None:
            genotypes = [x[:2] for x in variant.genotypes]		# Lấy 2 phần tử của 1 biến genotypes 
            for i in range(len(genotypes)):
                data[index[i] + '_1'].append(genotypes[i][0])
                data[index[i] + '_2'].append(genotypes[i][1])
            n_rows += 1
            # header: chr_pos_ref_alt
            headers.append(variant.CHROM + '_' + str(variant.POS) \
                + '_' + variant.REF + '_' + variant.ALT[0])		# Thêm các thông tin cần thiết vào header
        elif variant.POS < ref_position[idx]:
            continue
        elif variant.POS == ref_position[idx]:
            genotypes = [x[:2] for x in variant.genotypes]
            for i in range(len(genotypes)):
                data[index[i] + '_1'][idx] = genotypes[i][0]
                data[index[i] + '_2'][idx] = genotypes[i][1]
            n_rows += 1
            headers.append(variant.CHROM + '_' + str(variant.POS) \
                + '_' + variant.REF + '_' + variant.ALT[0])
            idx += 1
        else:
            while idx < len(ref_position) - 1 and variant.POS > ref_position[idx]:
                headers.append(variant.CHROM + '_' + str(ref_position[idx]) \
                    + '_' + 'nan' + '_' + 'nan')
                idx += 1
            if variant.POS == ref_position[idx]:
                genotypes = [x[:2] for x in variant.genotypes]
                for i in range(len(genotypes)):
                    data[index[i] + '_1'][idx] = genotypes[i][0]
                    data[index[i] + '_2'][idx] = genotypes[i][1]
                n_rows += 1
                headers.append(variant.CHROM + '_' + str(variant.POS) \
                    + '_' + variant.REF + '_' + variant.ALT[0])
                idx += 1
        if n_rows >= limit_rows:
            break
        
    while idx < len(ref_position):
        headers.append(variant.CHROM + '_' + str(ref_position[idx]) \
            + '_' + 'nan' + '_' + 'nan')
        idx += 1
        
    print('Number of rows: {}'.format(n_rows))
    df = pd.DataFrame(data, index=headers).T		# Tạo một dataframe với header và dữ liệu thu được ở trên
    
    if saved:
        out_path = path.replace('.vcf.gz', '.csv')
        df.to_csv(out_path)
    return df

def shuffle_data(dataset):
    '''
    dataset: dict
    return: dict
    '''
    shuffle_index = np.random.permutation(len(dataset['data']))		# Tạo một hoán vị của các số nguyên đầu tiên 
    dataset['data'] = np.array(dataset['data'])[shuffle_index]
    dataset['label'] = np.array(dataset['label'])[shuffle_index]
    return dataset

def split_dataset(dataset, split_ratio, shuffle=True):
    '''
    dataset: dict
    return: list of dict
    '''
    if shuffle:
        dataset = shuffle_data(dataset)		# Đảo thứ tự dữ liệu nếu trong lệnh có gọi shuffle
        
    split_point = int(len(dataset['data']) * split_ratio)		# Chọn điểm mà tại đó bắt đầu tách thành tập train và test
    train_dataset = {
        'data': dataset['data'][:split_point],
        'label': dataset['label'][:split_point]
    }
    test_dataset = {
        'data': dataset['data'][split_point:],
        'label': dataset['label'][split_point:]
    }
    return train_dataset, test_dataset

def save_to_bin(dataset, path):
    '''
    dataset: dict
    path: path to save file
    '''
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print('Saved to {}'.format(path))

def load_from_bin(path):
    '''
    path: path to load file
    return: dict
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)