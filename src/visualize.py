
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
    
def save_train_val_losses(train_losses, val_losses, fold, model_name, hla_types, out_dir='output/train_losses', display=False):
    fig=plt.figure()
    plt.plot(train_losses,"-b",  label="train_losses") 
    plt.plot(val_losses,"-r", label="val_losses") 
    # plt.show()
    out_dir = 'output/train_val_losses'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig("{}/{}_{}".format(out_dir, model_name, 'train_val_losses_fold_'+str(fold)+'.png'))

def save_val_acc(train_losses, val_accuracyx, fold, model_name, hla_types, out_dir='output/train_losses', display=False):
    fig=plt.figure()
    name=hla_types
    np_accu=np.array(val_accuracyx).T
    for j in range(len(name)):
        plt.plot(np_accu[j],label=name[j])
    # plt.show()
    out_dir = 'output/train_val_acc'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig("{}/{}_{}".format(out_dir, model_name, 'train_val_acc_fold_'+str(fold)+'.png'))
    
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