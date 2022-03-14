
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
    
def save_train_valid_losses(train_losses, valid_losses, fold, model_name, out_dir='output/train_losses', display=False):
    fig=plt.figure()
    plt.plot(train_losses,"-b",  label="train_losses") 
    plt.plot(valid_losses,"-r", label="valid_losses") 
    plt.legend(loc="upper right")
    # plt.show()
    out_dir = 'output/train_valid_losses'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig("{}/{}_{}".format(out_dir, model_name, 'train_valid_losses_fold_'+str(fold)+'.png'))

def save_valid_acc(train_losses, valid_accuracy_epoch, fold, model_name, out_dir='output/train_losses', display=False):
    fig=plt.figure()
    name=['HLA_A','HLA_B','HLA_C','HLA_DQA1','HLA_DQB1','HLA_DRB1','HLA_DPB1']
    np_accu=np.array(valid_accuracy_epoch).T
    for j in range(len(name)):
        plt.plot(np_accu[j],label=name[j])
    plt.legend(loc="upper right")
    # plt.show()
    out_dir = 'output/train_valid_acc'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig("{}/{}_{}".format(out_dir, model_name, 'train_valid_acc_fold_'+str(fold)+'.png'))
    
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
            
def plot_box_plot(accuracy_folds):
    mean = np.mean(accuracy_folds, axis = 0)
    std = np.std(accuracy_folds,axis=0)
    name=['HLA_A','HLA_B','HLA_C','HLA_DQA1','HLA_DQB1','HLA_DRB1','HLA_DPB1']
    fig = plt.figure()
    plt.errorbar(name,mean,std,ls='none')
    plt.show()
    out_dir = 'output/evaluation'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(out_dir+"/Evaluation_Metric.png")