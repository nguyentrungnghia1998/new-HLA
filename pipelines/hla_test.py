import argparse

import numpy as np
import torch as T
from src.collection import collection
from src.preprocess_data import pretrain
from models.SharedNet1C import SharedNet1C
from torch.autograd import Variable  
from objects.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, # default=None,
                        default='input/GSAv3_24.GRCh38.rename.HLAregion.vcf.gz',
                        help='path to dataset')
    parser.add_argument('--ref-position-path', type=str, # default=None, 
                        default='input/consensus23.phased.HLA.position.list',
                        help='path to index file')
    parser.add_argument('--position-path', type=str, # default=None, 
                        default='input/GSAv3_24.GRCh38.rename.HLAregion.position.list',
                        help='path to index file')
    parser.add_argument('--index-path', type=str, # default=None, 
                        default='input/GSAv3_24.GRCh38.rename.HLAregion.sample.list',
                        help='path to index file')
    parser.add_argument('--model-path', type=str, # default=None,
                        default='trainned_models/single_train/SharedNet2C_DPB1/SharedNet2C_model.pt',
                        help='path to model file')
    parser.add_argument('--co-model-path', type=str, # default=None,
                        default='trainned_models/multi_train/SharedNet2C_DPB1/SharedNet2C_model.pt',
                        help='path to model file')
    parser.add_argument('--label-path', type=str, # default=None,
                        default='input/DGV4VN_1015.HISAT_result.csv',
                        help='path to label file')
    parser.add_argument('--hla-types', type=str, default='DPB1',
                        help='comma separated list of hla alleles to be used for training, \
                        e.g. A,B,C,DQA1,DQB1,DRB1,DPB1')
    parser.add_argument('--output-path', type=str, default='output')
    args = parser.parse_args() 
    return args

def transform_dataset(dataset):
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    T_inputs = Variable(T.FloatTensor(np.array(dataset['data'])\
        .astype(np.float64)).to(device), requires_grad=False)
    T_targets = Variable(T.FloatTensor(np.array(dataset['label'])\
        .astype(np.float64)).to(device), requires_grad=False)
    return T_inputs, T_targets

def main():
    args = parse_args()
    dataset = pretrain( data_path=args.dataset_path, 
                        ref_pos_path=args.ref_position_path,
                        pos_path=args.position_path,
                        index_path=args.index_path, 
                        label_path=args.label_path,
                        hla_types=args.hla_types.split(','),
                        saved=True)
    
    ''' collect single column data and label using the trained model '''
    dataset = collection(dataset_path=args.dataset_path,
                        dataset=dataset,
                        model=model,
                        accept_threshold=args.accept_threshold)
    
    model = SharedNet1C(dataset['input-size'][1], dataset['outputs-size'])
    dataset = transform_dataset(dataset)
    evaluator = Evaluator(model=model)
    results = evaluator.evaluate(dataset)
    
if __name__ == '__main__':
    main()