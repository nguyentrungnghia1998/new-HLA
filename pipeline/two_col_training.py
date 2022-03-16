from src.data_helper import *
from models.SharedNet2C import SharedNet2C
from sklearn.model_selection import StratifiedKFold
from src.visualize import *
from src.trainer import Trainer

def trainning(dataset_path=None, dataset=None, optimizer=None, loss=None, 
             epochs=False, lr=None, batch_size=True, num_folds=None,
             n_repeats=None, print_every=None, save_every=None,
             output_path=None, save_dir=None, verbose=False):
    
    if dataset is None:
        ''' load csv file '''
        dataset = load_from_bin(dataset_path)
    
    if dataset['type'] != "2C":
       raise ValueError("Wrong dataset type, 2C is expected. Got {}".format(dataset['type']))
   
    trainset, testset = split_dataset(dataset, 0.9, shuffle=True)
    
    print('input_size: {}, output_size: {}'.format(dataset['input-size'], dataset['outputs-size']))
    
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    sample_feature_2D = np.reshape(trainset['data'],(trainset['data'].shape[0], -1))
    label_1D = np.zeros((trainset['label'].shape[0], 1))

    trainer = Trainer(
        loss=loss,
        optimizer=optimizer,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        n_repeats=n_repeats,
        print_every=print_every,
        save_every=save_every,
        save_dir=save_dir,
        verbose=verbose
        )
    
    print('-----------------------------------------------------')
    
    accuracy_folds = []
    for iter, (train_idx, val_idx) in enumerate(list(kfold.split(sample_feature_2D, label_1D))):
        fold = iter + 1
        print("Fold {}/{}".format(fold, num_folds))

        trainset_kfold = {}
        trainset_kfold['data'] = trainset['data'][train_idx]
        trainset_kfold['label'] = trainset['label'][train_idx]
        
        valset_kfold = {}
        valset_kfold['data'] = trainset['data'][val_idx]
        valset_kfold['label'] = trainset['label'][val_idx]

        trainer.model = SharedNet2C(dataset['input-size'], dataset['outputs-size'])
        trainer.train_loader = trainset_kfold
        trainer.test_loader = valset_kfold
        trainer.fold = fold
        trainer.train()

        print("Accuracy of fold ", fold, ":", trainer.valid_acc)
        save_acc(output_path, trainer.valid_acc, "Accuracy of fold " + str(fold) +": ")
        accuracy_folds.append(trainer.valid_acc)
        fold +=1
        
    print('-----------------------------------------------------')

    print("\nAverage accuracy:  ", np.mean(accuracy_folds, axis=0))
    print("\nStandart variation: ",np.std(accuracy_folds, axis=0))
    save_acc(output_path, np.mean(accuracy_folds, axis = 0), "\nAverage accuracy:  ")
    save_acc(output_path, np.std(accuracy_folds, axis = 0), "\nStandart variation:  ")
    plot_box_plot(accuracy_folds)
    
    print('-----------------------------------------------------')
    trainer.model = SharedNet2C(dataset['input-size'], dataset['outputs-size'])
    trainer.train_loader = trainset
    trainer.test_loader = testset
    trainer.fold = -1
    trainer.train()
    trainer.test()
    print("\nAccuracy of the whole dataset: ", trainer.valid_acc)
    save_acc(output_path, trainer.valid_acc, "\nAccuracy of the whole dataset: ")
    print('-----------------------------------------------------')
    
