import argparse
from src.preprocess_data import pretrain

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, # default=None,
                        default='input/GSAv3_24.GRCh38.rename.HLAregion.vcf.gz',
                        help='path to dataset')
    parser.add_argument('--index-path', type=str, # default=None, 
                        default='input/GSAv3_24.GRCh38.rename.HLAregion.list',
                        help='path to index file')
    parser.add_argument('--label-path', type=str, # default=None,
                        default='input/DGV4VN_1015.HISAT_result.csv',
                        help='path to label file')
    parser.add_argument('--hla-types', type=str, default='A',
                        help='comma separated list of hla alleles to be used for training, \
                        e.g. A,B,C,DQA1,DQB1,DRB1,DPB1')
    parser.add_argument('--output-path', type=str, default='output')
    args = parser.parse_args() 
    return args

def main():
    args = parse_args()
    dataset = pretrain( data_path=args.dataset_path, 
                        index_path=args.index_path, 
                        label_path=args.label_path,
                        hla_types=args.hla_types.split(','),
                        saved=True)
    print(dataset)
    
if __name__ == '__main__':
    main()