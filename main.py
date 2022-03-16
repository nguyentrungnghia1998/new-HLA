
import argparse
from pipeline.two_col_training import trainning as trainning_2c
from pipeline.one_col_training import trainning as trainning_1c
from pipeline.preprocess_data import pretrain

'''
python3 main.py --pipeline train2c --sample 20000 --model-name model.pt --epochs 5

'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, default='train2c')
    parser.add_argument('--data-path', type=str, default='input/consensus23.phased.HLA.vcf.gz')
    parser.add_argument('--dataset-path', type=str, default='input/consensus23.phased.HLA.2C.A.bin')
    parser.add_argument('--index-path', type=str, default='input/test.list')
    parser.add_argument('--label-path', type=str, default='input/DGV4VN_1006.Kourami_result.nonsort.csv')
    parser.add_argument('--sample', type=int, default=2009)
    parser.add_argument('--colapsed', action='store_true')
    parser.add_argument('--hla-types', type=str, default='A',
                        help='comma separated list of hla alleles to be used for training, \
                        e.g. A,B,C,DQA1,DQB1,DRB1,DPB1')
    parser.add_argument('--model-name', type=str, default='model.pt')
    parser.add_argument('--output-path', type=str, default='output')
    parser.add_argument('-l', '--loss', type=str, default='bce')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-k', '--k-fold', type=int, default=10)
    parser.add_argument('-e', '--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-r', '--n-repeats', type=int, default=1)
    parser.add_argument('-p', '--print-every', type=int, default=5)
    parser.add_argument('-s', '--save-every', type=int, default=100)
    parser.add_argument('-d', '--save-dir', type=str, default='./trainned_models')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args() 
    return args

def main():
    args = parse_args()
    pipelines = args.pipeline.split(',')
    dataset = None
    if 'pretrain' in pipelines:
        dataset = pretrain(data_path=args.data_path, 
                            index_path=args.index_path, 
                            label_path=args.label_path,
                            hla_types=args.hla_types.split(','),
                            colapsed=args.colapsed,
                            nrows=args.sample,
                            saved=True)
        
    elif 'train2c' in pipelines:
        trainning_2c(dataset_path=args.dataset_path,
                    dataset=dataset,
                    optimizer=args.optimizer,
                    loss=args.loss,
                    num_folds=args.k_fold,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    n_repeats=args.n_repeats,
                    print_every=args.print_every,
                    save_every=args.save_every,
                    save_dir=args.save_dir,
                    output_path=args.output_path,
                    verbose=args.verbose)
        
    elif 'train1c' in pipelines:
        trainning_1c(args)
    
if __name__ == '__main__':
    main()