from src.data_helper import *
from models.SharedNet2C import SharedNet2C
import argparse		# Thư viện giúp tạo định nghĩa command line trong terminal
import os
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib

from src.visualize import *
matplotlib.use('AGG')
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='input/consensus23.phased.HLA.bin')
    parser.add_argument('--model-name', type=str, default='model.pt')
    parser.add_argument('--output-path', type=str, default='output')
    parser.add_argument('-l', '--loss', type=str, default='bce')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-k', '--k-fold', type=int, default=10)
    parser.add_argument('-e', '--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-n', '--n-repeats', type=int, default=1)
    parser.add_argument('-p', '--print-every', type=int, default=5)
    parser.add_argument('-s', '--save-every', type=int, default=100)
    parser.add_argument('-d', '--save-dir', type=str, default='./trainned_models')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--sample', type=int, default=10009)
    args = parser.parse_args() 
    return args

def main():
    args = parse_args()
    ''' load csv file '''
    dataset = load_from_bin(args.data_path.replace('.vcf.gz', '.bin'))
    
    if dataset['type'] != "2C":
       raise ValueError("Wrong dataset type, 2C is expected. Got {}".format(dataset['type']))
   
    trainset, testset = split_dataset(dataset, 0.9, shuffle=True)
    
    print('input_size: {}, output_size: {}'.format(dataset['input-size'], dataset['outputs-size']))
    
    kfold = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=42)

    sample_feature_2D = np.reshape(trainset['data'],(trainset['data'].shape[0], -1))
    label_1D = np.zeros((trainset['label'].shape[0], 1))
    print('-----------------------------------------------------')
    fold = 1
    accuracy_folds = []
    for train_idx, val_idx in list(kfold.split(sample_feature_2D, label_1D)):
        print("Fold {}/{}".format(fold, args.k_fold))

        trainset_kfold = {}
        trainset_kfold['data'] = trainset['data'][train_idx]
        trainset_kfold['label'] = trainset['label'][train_idx]
        
        valset_kfold = {}
        valset_kfold['data'] = trainset['data'][val_idx]
        valset_kfold['label'] = trainset['label'][val_idx]

        trainer = Trainer(
            model=SharedNet2C(dataset['input-size'], dataset['outputs-size']),
            train_loader=trainset_kfold,
            test_loader=valset_kfold,
            loss=args.loss,
            optimizer=args.optimizer,
            fold=fold,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            n_repeats=args.n_repeats,
            print_every=args.print_every,
            save_every=args.save_every,
            save_dir=args.save_dir,
            verbose=args.verbose
            )
        trainer.train()

        print("Accuracy of fold ", fold, ":", trainer.valid_acc)
        save_acc(args.output_path, trainer.valid_acc, "Accuracy of fold " + str(fold) +": ")
        # _,_,precision,recall = trainer.test()
        # save_precision(args.output_path, precision,"Precision of fold " + str(fold) +": ")
        # save_recall(args.output_path, recall,"Recall of fold "+ str(fold)+": ")
        accuracy_folds.append(trainer.valid_acc)
        fold +=1

    print("\nAverage accuracy:  ", np.mean(accuracy_folds, axis=0))
    print("\nStandart variation: ",np.std(accuracy_folds, axis=0))
    save_acc(np.mean(accuracy_folds, axis = 0), "\nAverage accuracy:  ")
    save_acc(np.std(accuracy_folds, axis = 0), "\nStandart variation:  ")
    plot_box_plot(accuracy_folds)
    
if __name__ == "__main__":
    main()
