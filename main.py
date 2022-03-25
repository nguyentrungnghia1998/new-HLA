
import argparse
from pipeline.collection import collection
from pipeline.trainning import trainning
from pipeline.preprocess_data import pretrain

'''
<<<<<<< HEAD
python3 main.py --pipeline pretrain,train-1,train-2 \
    --data-path input/consensus23.phased.HLA.vcf.gz \
        --index-path 'input/test.list' \
            --label-path 'input/DGV4VN_1006.Kourami_result.nonsort.csv' \
                --sample 10009 -l bce -o adam -k 10 -e 2 --lr 0.007 -b 64 --hla 'A' -e 10 -v
=======
python3 main.py --pipeline pretrain,train --data-path 'input/consensus23.phased.HLA.bin' \
    --index-path 'input/test.list' --label-path 'input/DGV4VN_1006.Kourami_result.nonsort.csv' \
    --sample 50009 --model-name 'model.pt' --output-path 'output' \
    -l bce -o adam -k 10 -e 2 -lr 0.007 -b 64 -n 1 -p 5 -s 100 -d ./trainned_models -v
>>>>>>> d7b334d18d81c61b5fb06ce1c37f593ace7e832e
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, default='pretrain,train')
    parser.add_argument('--data-path', type=str, default='input/consensus23.phased.HLA.vcf.gz')
    parser.add_argument('--dataset-path', type=str, default='input/consensus23.phased.HLA.2C.A.bin')
    parser.add_argument('--index-path', type=str, default='input/test.list')
    parser.add_argument('--label-path', type=str, default='input/DGV4VN_1006.Kourami_result.nonsort.csv')
    parser.add_argument('--sample', type=int, default=5009)
    parser.add_argument('--colapsed', action='store_true')
    parser.add_argument('--hla-types', type=str, default='A',
                        help='comma separated list of hla alleles to be used for training, \
                        e.g. A,B,C,DQA1,DQB1,DRB1,DPB1')
    parser.add_argument('--model-path', type=str, default='./trainned_models/multi_train/SharedNet2C/SharedNet2C_model.pt')
    parser.add_argument('--output-path', type=str, default='output')
    parser.add_argument('-l', '--loss', type=str, default='bce')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('--use_cross_validation', action='store_true')
    parser.add_argument('-k', '--k-fold', type=int, default=2)
    parser.add_argument('-e', '--epochs', type=int, default=1)
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
    model = None
    if 'pretrain' in pipelines:
        dataset = pretrain( data_path=args.data_path, 
                            index_path=args.index_path, 
                            label_path=args.label_path,
                            hla_types=args.hla_types.split(','),
                            colapsed=args.colapsed,
                            nrows=args.sample,
                            saved=True)
        
    if 'train-1' in pipelines:
        model = trainning(dataset_path=args.dataset_path,
                            dataset=dataset,
                            optimizer=args.optimizer,
                            loss=args.loss,
                            use_cross_validation=args.use_cross_validation,
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
        
        dataset = collection(dataset_path=args.dataset_path,
                            dataset=dataset,
                            model=model, model_path=args.model_path)
        
    if 'train-2' in pipelines:
        trainning(dataset_path=args.dataset_path,
                    dataset=dataset,
                    optimizer=args.optimizer,
                    loss=args.loss,
                    num_folds=args.k_fold,
                    epochs=args.epochs,
                    lr=args.lr,
                    use_cross_validation=args.use_cross_validation,
                    batch_size=args.batch_size,
                    n_repeats=args.n_repeats,
                    print_every=args.print_every,
                    save_every=args.save_every,
                    save_dir=args.save_dir,
                    output_path=args.output_path,
                    verbose=args.verbose)
    
if __name__ == '__main__':
    main()