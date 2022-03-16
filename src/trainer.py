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
                 device=T.device("cuda" if T.cuda.is_available() else "cpu"), lr=0.0005, epochs=200, batch_size=64,
                 n_repeats = 2, print_every=1, save_every=500,
                 save_dir="./trainned_models",
                 save_name="model.pt", verbose=True):
        self.model = model
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader          # Khai báo biến load của 2 tập train và test
        self.fold = fold 
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_repeats = n_repeats              # Số lần lặp lại quá trình học
        self.print_every = print_every
        self.save_every = save_every            # Số bước tuần hoàn thực hiện in kết quả và lưu mô hình
        self.save_dir = save_dir
        self.save_name = save_name
        self.verbose = verbose          # Chọn cách thức hiển thị kết quả sau mỗi vòng lặp 
        self.train_losses = []
        self.hla_types = []

        self.valid_losses = []
        self.test_losses = []           # Tính loss của các tập train, test, validation mỗi epoch 
        self.train_acc = []
        self.valid_acc = []
        self.valid_accuracy_epoch=[]
        self.test_acc = []
        
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
            inputs = []
            targets = []
            for i in range(len(dataset['data'])):
                row = dataset['data'][i]
                row_1 = [row[0], row[0]]
                row_2 = [row[1], row[1]]
                inputs.extend([row_1, row_2])
                targets.extend([dataset['label'][i], dataset['label'][i]])
                
            T_input_batch = Variable(T.FloatTensor(np.array(inputs)\
                .astype(np.float64)).to(self.device), requires_grad=False)
            T_target_batch = Variable(T.FloatTensor(np.array(targets)\
                .astype(np.float64)).to(self.device), requires_grad=False)
                
            return (T_input_batch, T_target_batch)
    
    def train(self):
        self.hla_types = [out[0] for out in self.model.outputs_size]
        if self.fold >= 0:
            model_path = 'trainned_models/' + self.model.name + '/fold_' + str(self.fold) + '/' + \
                '_'.join(self.hla_types)
        else:
            self.model.name = self.model.name + '.' + '_'.join(self.hla_types)
        self.model.set_loss_function(self.loss)
        self.model.set_optimizer(self.optimizer, self.lr)         # Chon hàm loss, model, learning rate và thuật toán tối ưu
        
        self.model.to(self.device)
        self.model._train()              # Chuyển model về gpu nếu có, và xác lập chế độ train 
        for iter in range(self.epochs):
            train_batches = self.transform_dataset(self.train_loader, mode='train', batch_size=self.batch_size, shuffle=True)
            t = tqdm(train_batches, desc="Epochs {}".format(iter))              # Khai báo 1 tiến trình tqdm cho từng batch 
            losses = 0
            for batch_idx, (inputs, targets) in enumerate(t):           # Quét vòng lặp đến tất cả dữ liệu trong batch hiện ta
                self.model.reset_grad()
                output = self.model(inputs)
                loss = self.model.loss(output, targets.detach())                # Tính hàm loss giữa đầu ra output và targets
                loss.backward()
                self.model.step()               # Cập nhật tham số mạng nơ ron ở cuối mô hình 
                losses+=loss.item()
                t.set_postfix(loss=loss.item())         # Đặt hiển thị loss ở cuối thanh tiến trình n 

                if batch_idx % self.save_every == 0 and batch_idx != 0 or batch_idx == len(train_batches) - 1:
                    self.model.save(path=model_path)
                    self.model.save_train_losses(self.train_losses)
                    self.valid_acc, valid_loss = self.test()            # Trả về kết quả accuracy từng batch của tập test
                    t.set_postfix(val_acc=self.valid_acc)
                    self.model._train()

            losses = losses/len(train_batches)
            print("\nValid_loss: ", valid_loss)
            print('Train_loss: ',losses)
            self.train_losses.append(losses)
            self.valid_losses.append(valid_loss)
            self.valid_accuracy_epoch.append(self.valid_acc)
            save_train_valid_losses(self.train_losses, self.valid_losses, fold=self.fold,
                            model_name=self.model.name, hla_types=self.hla_types)
            save_valid_acc(self.train_losses, self.valid_accuracy_epoch, fold=self.fold,
                           model_name=self.model.name, hla_types=self.hla_types)
       
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
                output = self.model.predict(data)
                presize = 0
                for name, output_size in self.model.outputs_size:
                    allele_outs = output[presize:presize + output_size].argsort()[-2:][::-1]
                    allele_target = target[presize:presize + output_size].cpu().numpy().argsort()[-2:][::-1]
                    if ((allele_outs[0] == allele_target[0]) or \
                        (allele_outs[0] == allele_target[1] and \
                        target[presize:presize + output_size].cpu().numpy()[allele_target[1]]==1)):
                        accuracies[name] += 1
                    val_losses[name] += self.model.loss(T.FloatTensor(output[presize:presize + output_size]), T.FloatTensor(target[presize:presize + output_size])).item()
                    presize += output_size
        return [np.round(acc / len(self.test_loader['data']) / 2, 2) for acc in accuracies.values()], np.mean(list(val_losses.values()))/(_iter+1)
