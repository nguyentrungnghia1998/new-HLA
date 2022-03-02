from src.data_helper import *
from models.SharedNet2D import SharedNet2D
import argparse		# Thư viện giúp tạo định nghĩa command line trong terminal

from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='input/test_subsample.vcf.gz')
    parser.add_argument('--model-name', type=str, default='model.pt')
    parser.add_argument('--output-path', type=str, default='output')
    parser.add_argument('-l', '--loss', type=str, default='bce')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-n', '--n-repeats', type=int, default=1)
    parser.add_argument('-p', '--print-every', type=int, default=5)
    parser.add_argument('-s', '--save-every', type=int, default=10)
    parser.add_argument('-d', '--save-dir', type=str, default='./trainned_models')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--sample', type=int, default=10009)
    args = parser.parse_args() 
    return args

def main():
    args = parse_args()
    ''' load csv file '''
    dataset = load_from_bin(args.data_path.replace('.vcf.gz', '.bin'))
    
    trainset, testset = split_dataset(dataset, 0.8, shuffle=True)
    
    print('input_size: {}, output_size: {}'.format(dataset['input-size'], dataset['outputs-size']))
    
    trainer = Trainer(
        model=SharedNet2D(dataset['input-size'], dataset['outputs-size']),
        train_loader=trainset,
        test_loader=testset,
        loss=args.loss,
        optimizer=args.optimizer,
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

if __name__ == "__main__":
    main()
