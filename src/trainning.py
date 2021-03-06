''' 
    Author: Vu Quoc Hien
    Date created: 2022-03-14
'''

from models.SharedNet2C import SharedNet2C
from src.data_helper import *
from objects.evaluator import Evaluator
from models.SharedNet1C import SharedNet1C
from sklearn.model_selection import StratifiedKFold
from src.visualize import *
from objects.single_trainer import Trainer as SingleTrainer
from objects.multi_trainer import Trainer as MultiTrainer

def trainning(dataset_path=None, dataset=None, optimizer=None, loss=None, 
             epochs=False, lr=None, batch_size=None, use_cross_validation=True, 
             num_folds=None, split_ratio=0.9, using_collection=False,
             output_path=None, save_dir=None, verbose=False):
    
    if dataset is None:
        ''' load csv file '''
        dataset = load_from_bin(dataset_path)
        
    ''' Validate dataset '''
    if dataset['type'] == "2C":
        Model = SharedNet2C
        Trainer = MultiTrainer
    elif dataset['type'] == "1C":
        Model = SharedNet1C
        Trainer = SingleTrainer
    else:
       raise ValueError("Wrong dataset type, 1C/2C is expected. Got {}".format(dataset['type']))
    
    if using_collection:
        trainset, testset = split_dataset(dataset, split_ratio, shuffle=True)
        #testset = trainset
    else:
        trainset, testset = split_dataset(dataset, split_ratio, shuffle=True)
    
    print('input_size: {}, output_size: {}'.format(dataset['input-size'], dataset['outputs-size']))
    print('train_size: {}, test_size: {}'.format(len(trainset['data']), len(testset['data'])))
    
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    sample_feature = np.reshape(trainset['data'],(trainset['data'].shape[0], -1))
    labels = np.zeros((trainset['label'].shape[0], 1))
    
    trainer = Trainer(
        loss=loss,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_dir=save_dir,
        verbose=verbose
        )
    
    print('-----------------------------------------------------')
    if use_cross_validation:
        accuracy_folds = []
        trainer.trainning_fold = True
        for iter, (train_idx, val_idx) in enumerate(list(kfold.split(sample_feature, labels))):
            fold = iter + 1
            print("Fold {}/{}".format(fold, num_folds))

            trainset_kfold = {}
            trainset_kfold['data'] = trainset['data'][train_idx]
            trainset_kfold['label'] = trainset['label'][train_idx]
            
            valset_kfold = {}
            valset_kfold['data'] = trainset['data'][val_idx]
            valset_kfold['label'] = trainset['label'][val_idx]

            trainer.set_model(Model(dataset['input-size'], dataset['outputs-size']))
            trainer.set_dataset(trainset_kfold, valset_kfold)
            trainer.fold = fold
            trainer.train()

            save_acc(output_path, trainer.val_accuracy[-1], "Accuracy of fold " + str(fold) +": ")
            accuracy_folds.append(trainer.val_accuracy[-1])
            fold += 1
            
        print('-----------------------------------------------------')
        print("\nAverage accuracy:  ", np.mean(accuracy_folds, axis=0))
        print("\nStandart variation: ",np.std(accuracy_folds, axis=0))
        save_acc(output_path, np.mean(accuracy_folds, axis = 0), "\nAverage accuracy:  ")
        save_acc(output_path, np.std(accuracy_folds, axis = 0), "\nStandart variation:  ")
        plot_box_plot(accuracy_folds, [hla[0] for hla in trainer.model.outputs_size])
        print('-----------------------------------------------------')
    
    trainer.trainning_fold = False
    trainer.set_model(Model(dataset['input-size'], dataset['outputs-size']))
    trainer.set_dataset(trainset, testset)
    trainer.train()
    trainer.model.save(path=trainer.model_path)
    val_acc, val_loss = trainer.test()
    print("Validation accuracy of the whole dataset: ", val_acc)
    print("Validation loss of the whole dataset: ", val_loss)
    save_acc(output_path, val_acc, "\nAccuracy of the whole dataset: ")
    print('-----------------------------------------------------')

    if dataset['type']=="1C":
        evaluation=Evaluator(trainer.model)
        test_batch=trainer.get_test_batch()
        metric=evaluation.evaluate(test_batch,metric_names=['confusion_matrix'])
        print(metric)
    return trainer.model
    
