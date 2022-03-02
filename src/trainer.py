import os		# Thư viện làm việc với đường dẫn và file trong python
import torch as T
import numpy as np
from torch.autograd import Variable		# Hàm khai báo các biến tensor và có thể thay đổi gradient của nó
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.data_helper import shuffle_data

class Trainer:
    def __init__(self, model, loss, optimizer, train_loader=None, test_loader=None,
                 device=T.device("cpu"), lr=0.0005, epochs=200, batch_size=64,
                 n_repeats = 2, print_every=1, save_every=500, 
                 save_dir="./trainned_models",
                 save_name="model.pt", verbose=True):
        self.model = model
        self.model.set_loss_function(loss)
        self.model.set_optimizer(optimizer, lr)		# Khai báo hàm loss, model, learning rate và thuật toán tối ưu
        self.train_loader = train_loader
        self.test_loader = test_loader		# Khai báo biến load của 2 tập train và test
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_repeats = n_repeats		# Số lần lặp lại quá trình học
        self.print_every = print_every
        self.save_every = save_every		# Số bước tuần hoàn thực hiện in kết quả và lưu mô hình
        self.save_dir = save_dir
        self.save_name = save_name
        self.verbose = verbose		# Chọn cách thức hiển thị kết quả sau mỗi vòng lặp 
        self.train_losses = []

        self.valid_losses = []
        self.test_losses = []		# Tính loss của các tập train, test, validation mỗi epoch 
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []
        
    def split_batch(self, dataset, batch_size=None, shuffle=False):
        """
        Split dataset into batches
        :param dataset: dataset
        :param batch_size: batch size
        :return: batches
        """
        if shuffle:
            dataset = shuffle_data(dataset)		# Nếu gọi biến shuffle thì trộn bộ dữ liệu đã cho
        batches = []
        if batch_size is None:
            batch_size = len(dataset['data'])		# Nếu không khai báo batch_size thì coi như là huấn luyện toàn bộ dữ liệu
        for i in range(0, len(dataset['data']), batch_size):
            batches.append((dataset['data'][i:i + batch_size], dataset['label'][i:i + batch_size]))
        return batches
    
    def train(self):
        self.model.to(self.device)
        self.model.train()		# Chuyển model về gpu nếu có, và xác lập chế độ train 
        trainset = self.train_loader
        for iter in range(self.epochs):
            train_batches = self.split_batch(trainset, self.batch_size, shuffle=True)
            t = tqdm(train_batches, desc="Iter {}".format(iter))		# Khai báo 1 tiến trình tqdm cho từng batch 
            for batch_idx, (data, targets) in enumerate(t):		# Quét vòng lặp đến tất cả dữ liệu trong batch hiện ta
                inputs = Variable(T.FloatTensor(np.array(data).astype(np.float64)).to(self.device), requires_grad=True)
                targets = Variable(T.FloatTensor(np.array(targets).astype(np.float64)).to(self.device), requires_grad=True)
                self.model.reset_grad()
                output = self.model(inputs)
                loss = self.model.loss(output, targets.detach())		# Tính hàm loss giữa đầu ra output và targets
                loss.backward()
                self.train_losses.append(loss.item())
                self.model.step()		# Cập nhật tham số mạng nơ ron ở cuối mô hình 
                
                t.set_postfix(loss=loss.item())		# Đặt hiển thị loss ở cuối thanh tiến trình n 

                if batch_idx % self.save_every == 0 and batch_idx != 0 or batch_idx == len(train_batches) - 1:
                    self.save_train_losses()
                    self.model.save()
                    self.model.save_train_losses(self.train_losses)
                    returns = self.test()		# Trả về kết quả accuracy từng batch của tập test
                    t.set_postfix(val_acc=returns)
                    self.model._train()

            
    def load_model_from_path(self, path):
        self.model.load_state_dict(T.load(path))
    
    def save_train_losses(self):
        plt.plot(self.train_losses) 
        out_dir = 'output/train_losses'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig("{}/{}_{}".format(out_dir, self.model.name, 'train_losses.png'))
    
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
            t = tqdm(zip(self.test_loader['data'], self.test_loader['label']), desc="Testing")
            for _iter, (data, target) in enumerate(t):		# Lấy số vòng lặp, data và target lần lượt trong tqdm
                output = self.model.predict(data)
                presize = 0
                for name, output_size in self.model.outputs_size:
                    outs = output[presize:presize + output_size].argsort()[-2:][::-1]
                    targets = target[presize:presize + output_size].argsort()[-2:][::-1]
                    presize += output_size
                    if (outs[0] == targets[0]) \
                        or (outs[0] == targets[1]):
                        accuracies[name] += 1
                    # val_losses[name] += self.model.loss(output[presize:presize + output_size], target[presize:presize + output_size]).item()
        return [np.round(acc / len(self.test_loader['data']), 2) for acc in accuracies.values()]
