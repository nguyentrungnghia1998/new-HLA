import os
import matplotlib               # Thư viện làm việc với đường dẫn và file trong python
matplotlib.use('AGG')
import torch as T
import numpy as np
from torch.autograd import Variable             # Hàm khai báo các biến tensor và có thể thay đổi gradient của nó
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.visualize import *

from src.data_helper import shuffle_data

class Trainer:
    def __init__(self, model=None, loss='bce', optimizer='adam', train_loader=None, test_loader=None, fold=1,
                 device=None, lr=0.0005, epochs=200, batch_size=64,
                 save_dir="./trainned_models",
                 save_name="model.pt", verbose=True):
        self.model = model
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader          # Khai báo biến load của 2 tập train và test
        self.fold = fold 
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.save_name = save_name
        self.verbose = verbose          # Chọn cách thức hiển thị kết quả sau mỗi vòng lặp 
        self.train_losses = []
        self.hla_types = []

        self.val_losses = []
        self.test_losses = []           # Tính loss của các tập train, test, validation mỗi epoch 
        self.train_acc = []
        self.val_acc = []
        self.val_accuracy=[]
        self.test_acc = []
        self.trainning_fold = False
        self.model_path = None
        
    def set_model_path(self):
        self.model_path = os.path.join(self.save_dir, 'single_train', self.model.name + ('_'.join(self.hla_types)))
        
    def set_model(self, model):
        self.model = model
        self.model.set_loss_function(self.loss)
        self.model.set_optimizer(self.optimizer, self.lr)         # Chon hàm loss, model, learning rate và thuật toán tối ưu
        self.model.to(self.device)
        self.hla_types = [out[0].replace('HLA', '') for out in self.model.outputs_size]
        self.set_model_path()
    
    def set_dataset(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        
    def transform_dataset(self, dataset, mode='train', batch_size=64, shuffle=True):
        """
        Transform dataset to batchs
        :param dataset: dataset
        :param mode: train or test
        :return: transformed dataset
        """
        if mode == 'train':
            if shuffle:
                dataset = shuffle_data(dataset)         # Nếu gọi biến shuffle thì trộn bộ dữ liệu đã cho
            batches = []
            for i in range(0, len(dataset['data']), batch_size):
                input_batch = dataset['data'][i:i + batch_size]
                target_batch = dataset['label'][i:i + batch_size]
                T_input_batch = Variable(T.FloatTensor(np.array(input_batch)\
                    .astype(np.float64)).to(self.device), requires_grad=True)
                T_target_batch = Variable(T.FloatTensor(np.array(target_batch)\
                    .astype(np.float64)).to(self.device), requires_grad=True)
                batches.append((T_input_batch, T_target_batch))
            return batches
        elif mode == 'test':
            T_inputs = Variable(T.FloatTensor(np.array(dataset['data'])\
                .astype(np.float64)).to(self.device), requires_grad=False)
            T_targets = Variable(T.FloatTensor(np.array(dataset['label'])\
                .astype(np.float64)).to(self.device), requires_grad=False)
            return T_inputs, T_targets
    
    def train(self):
        if len(self.train_loader['data']) == 0:
            raise Exception('Train dataset is empty')
        
        self.model._train() # model to train mode, using batch norm and dropout
        
        for iter in range(self.epochs):
            train_batches = self.transform_dataset(self.train_loader, mode='train', 
                                                   batch_size=self.batch_size, shuffle=True)
            t = tqdm(train_batches, desc="Epochs {}".format(iter))              # Khai báo 1 tiến trình tqdm cho từng batch 
            losses = 0
            for batch_idx, (inputs, targets) in enumerate(t):           # Quét vòng lặp đến tất cả dữ liệu trong batch hiện ta
                self.model.reset_grad()
                output = self.model(inputs)
                loss = self.model.loss(output, targets.detach())                # Tính hàm loss giữa đầu ra output và targets
                loss.backward()
                self.model.step()               # Cập nhật tham số mạng nơ ron ở cuối mô hình 
                losses += loss.item()
                
            self.model.save(path=self.model_path)
            val_acc, val_loss = self.test()            # Trả về kết quả accuracy từng batch của tập test
            t.set_postfix(train_loss=loss.item(), val_loss=val_loss, val_acc=val_acc)
            
            if self.verbose:
                print('Epoch: {}/{}'.format(iter, self.epochs))
                print('Train loss: {:.4f} \t Val loss: {:.4f}'.format(loss.item(), val_loss))
                print('Val accuracy:', val_acc)
                
            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss)
            self.val_accuracy.append(val_acc)
            
            save_train_val_losses(self.train_losses, self.val_losses, self.trainning_fold, fold=self.fold,
                            model_name=self.model.name,data_type="Single_Trainer",hla_types=self.hla_types)
            save_val_acc(self.train_losses, self.val_accuracy, self.trainning_fold, fold=self.fold,
                           model_name=self.model.name,data_type="Single_Trainer",hla_types=self.hla_types)
        
    def test(self):
        if len(self.test_loader['data']) == 0:
            print('Skipping test')
        self.model._eval()
        accuracies = {}
        val_losses = {}
        for name, output_size in self.model.outputs_size:		# Quét vòng for đến tất cả các HLA và số output tương ứng
            accuracies[name] = 0
            val_losses[name] = 0
            
        with T.no_grad():		# Tắt gradient các tensor trong khối lệnh phía dưới 
            test_batches = self.transform_dataset(self.test_loader, mode='test')
            for _iter, (data, target) in enumerate(zip(test_batches[0], test_batches[1])):		# Lấy số vòng lặp, data và target lần lượt trong tqdm
                output = self.model(data).flatten(0)
                presize = 0
                for name, output_size in self.model.outputs_size:
                    allele_out = output[presize:presize + output_size].argsort()[-1]
                    allele_target = target[presize:presize + output_size].argsort()[-1]
                    if allele_out == allele_target:
                        accuracies[name] += 1
                    val_losses[name] += self.model.loss(output[presize:presize + output_size], 
                                                        target[presize:presize + output_size]).item()
                    presize += output_size
        
        return [np.round(acc / len(self.test_loader['data']), 2) for acc in accuracies.values()], \
            np.mean(list(val_losses.values()))/(_iter+1)
    
    def get_test_batch(self):
        test_batches= self.transform_dataset(self.test_loader,mode='test')
        return test_batches
    