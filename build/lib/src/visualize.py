
import os
from matplotlib import pyplot as plt
import numpy as np


def save_train_losses(train_losses, iter, model_name, out_dir='output/train_losses', display=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig = plt.figure()
    plt.plot(train_losses, label="train_losses")
    plt.legend(loc="upper right")
    if display:
        plt.show()
    fig.savefig("{}/{}_{}".format(out_dir, model_name, 'train_losses_fold_'+str(iter)+'.png'))
    
def save_train_val_losses(train_losses, val_losses, training_fold=None, fold=None, model_name=None, data_type=None,  hla_types=None, out_dir='output/train_losses', display=False):
    plt.plot(train_losses,"-b",  label="train_losses") 
    plt.plot(val_losses,"-r", label="valid_losses") 
    plt.legend(loc="upper right")
    if not training_fold:
        out_dir = 'output/'+data_type+'/'+model_name+'/FinalTrainingResults/HLA'+''.join(hla_types)
    else:
        out_dir = 'output/'+data_type+'/'+model_name+'/CrossValidationResults/Fold_'+str(fold)+'/HLA'+''.join(hla_types)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not training_fold:
        plt.savefig("{}/{}".format(out_dir, 'train_test_losses.png'))
    else:
        plt.savefig("{}/{}".format(out_dir, 'train_valid_losses.png'))
    plt.close()

def save_val_acc(train_losses, val_accuracy, training_fold=None, fold=None, model_name=None,data_type=None, hla_types=None, out_dir='output/train_losses', display=False):
    name=hla_types
    np_accu=np.array(val_accuracy).T
    for j in range(len(name)):
        plt.plot(np_accu[j],label='HLA'+name[j])
    plt.legend(loc="upper right")
    if not training_fold:
        out_dir = 'output/'+data_type+'/'+model_name+'/FinalTrainingResults/HLA'+''.join(hla_types)
    else:
        out_dir = 'output/'+data_type+'/'+model_name+'/CrossValidationResults/Fold_'+str(fold)+'/HLA'+''.join(hla_types)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not training_fold:
        plt.savefig("{}/{}".format(out_dir, 'test_accuracy.png'))
    else:
        plt.savefig("{}/{}".format(out_dir, 'validation_accuracy.png'))
    plt.close()
    
def save_acc(path, accuracy, name_acc):
    with open(path + "/kfold_acc_model_2D.txt", 'a') as f:
        f.writelines(name_acc + str(accuracy)+"\n")
        
def save_precision(path, precision, name_precision):
    with open(path + "/kfold_precision_model_2D.txt", 'a') as f:
        f.writelines(name_precision + "\n")
        for i in precision:
            f.writelines(i+":"+str(precision[i])+"\n")
            
def save_recall(path, recall, name_recall):
    with open(path + "/kfold_recall_model_2D.txt", 'a') as f:
        f.writelines(name_recall + "\n")
        for i in recall:
            f.writelines(i+":"+str(recall[i])+"\n")
            
def plot_box_plot(accuracy_folds, hla_types):
    mean = np.mean(accuracy_folds, axis = 0)
    std = np.std(accuracy_folds,axis=0)
    
    name=hla_types
    fig = plt.figure()
    plt.errorbar(name,mean,std,ls='none')
    # plt.show()
    out_dir = 'output/evaluation'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(out_dir+"/Evaluation_Metric.png")