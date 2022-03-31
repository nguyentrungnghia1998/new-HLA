import argparse
from src.collection import collection
from src.trainning import trainning
from src.preprocess_data import pretrain

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true',
                        help='pretrain model, preprocess data')
    parser.add_argument('--trainning', action='store_true', default=True,
                        help='trainning model')
    parser.add_argument('--dataset-path', type=str, # default=None,
                        default='input/consensus23.phased.HLA.2C.A.bin',
                        help='path to dataset')
    parser.add_argument('--index-path', type=str, # default=None, 
                        default='input/consensus23.phased.HLA.sample.list',
                        help='path to index file')
    parser.add_argument('--label-path', type=str, # default=None,
                        default='input/DGV4VN_1006.Kourami_result.nonsort.csv',
                        help='path to label file')
    parser.add_argument('--sample', type=int, default=10000,
                        help='number of samples to use')
    parser.add_argument('--accept-threshold', type=float, default=0.5,
                        help='threshold to accept for extract single column data')
    parser.add_argument('--hla-types', type=str, default='A',
                        help='comma separated list of hla alleles to be used for training, \
                        e.g. A,B,C,DQA1,DQB1,DRB1,DPB1')
    parser.add_argument('--output-path', type=str, default='output')
    parser.add_argument('-l', '--loss', type=str, default='bce',
                        help='loss function to use')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='optimizer to use')
    parser.add_argument('--use_cross_validation', action='store_true')
    parser.add_argument('-k', '--k-fold', type=int, default=10,
                        help='number of folds to use for cross validation')
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005,  
                        help='learning rate')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('-d', '--model-save-dir', type=str, 
                        default='./trainned_models',
                        help='directory to save model')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='verbose mode')
    args = parser.parse_args() 
    return args

def main():
    args = parse_args()
    dataset = None
    model = None
    
    ''' preprocess data '''
    if args.pretrain:
        dataset = pretrain( data_path=args.dataset_path, 
                            index_path=args.index_path, 
                            label_path=args.label_path,
                            hla_types=args.hla_types.split(','),
                            nrows=args.sample,
                            saved=True)
        
    ''' trainning model (two column input)'''
    if args.trainning:
        model = trainning(dataset_path=args.dataset_path,
                            dataset=dataset,
                            optimizer=args.optimizer,
                            loss=args.loss,
                            use_cross_validation=False,
                            num_folds=args.k_fold,
                            epochs=args.epochs,
                            lr=args.lr,
                            batch_size=args.batch_size,
                            save_dir=args.model_save_dir,
                            output_path=args.output_path,
                            using_collection=True,
                            verbose=args.verbose)
        
        ''' collect single column data and label using the trained model '''
        dataset = collection(dataset_path=args.dataset_path,
                            dataset=dataset,
                            model=model,
                            accept_threshold=args.accept_threshold)
        
        ''' trainning model (single column input) '''
        trainning(dataset_path=args.dataset_path,
                    dataset=dataset,
                    optimizer=args.optimizer,
                    loss=args.loss,
                    num_folds=args.k_fold,
                    epochs=args.epochs,
                    lr=args.lr,
                    use_cross_validation=args.use_cross_validation,
                    batch_size=args.batch_size,
                    save_dir=args.model_save_dir,
                    output_path=args.output_path,
                    verbose=args.verbose)
    
if __name__ == '__main__':
    main()