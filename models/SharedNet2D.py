#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn		# Module chứa các lớp trong mạng nơ-ron thường dùng
import torch.nn.functional as F
import random
import json		# Thư viện làm việc với file json, đọc và ghi dữ liệu json
import numpy as np
from torch import optim
from torchsummary.torchsummary import summary		# Tóm lại một mô hình dưới dạng 1 sơ đồ khối có đầu ra mỗi lớp

config = json.load(open('models/Model_2D.json'))

class PrivatedNet(nn.Module):
    def __init__(self, name, input_size, output_size):
        super(PrivatedNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fc1_len = config[name]["fc1_len"]		# Tìm số lớp ẩn của lớp fully connected thứ 1 trong mạng độc lập
        fc2_len = config[name]["fc2_len"]		
        self.fc1 = nn.Linear(input_size, fc1_len).to(self.device)		# Khai báo lớp tuyến tính cho từng HLA riêng.S
        self.bn1 = nn.BatchNorm1d(fc1_len).to(self.device)
        self.fc2 = nn.Linear(fc1_len, fc2_len).to(self.device)
        self.bn2 = nn.BatchNorm1d(fc2_len).to(self.device)
        self.fc3 = nn.Linear(fc1_len, output_size).to(self.device)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        # out = F.relu(self.bn2(self.fc2(out)))		# Thêm hàm relu để phi tuyến hóa kết quả sau lớp linear
        out = self.fc3(out)
        return torch.softmax(out, dim=1)
        

class SharedNet2D(nn.Module):
    def __init__(self, input_size, outputs_size):
        super(SharedNet2D, self).__init__()
        self.name = 'SharedNet'
        self.input_size = input_size		# Số đầu vào của mạng SharedNet chứa tất cả các loại HLA
        self.outputs_size = outputs_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conv1_num_filter = config['HLA_Shared']["conv1_num_filter"]		# Số bộ lọc của lớp tích chập đầu tiên
        conv2_num_filter = config['HLA_Shared']["conv2_num_filter"]
        conv1_kernel_size = config['HLA_Shared']["conv1_kernel_size"]
        conv2_kernel_size = config['HLA_Shared']["conv2_kernel_size"]		# Kích thước kernel của 2 lớp tích chập
        max_pool_size = config['HLA_Shared']["max_pool_size"]		# Kích thước lớp tích chập
        fc_len = config['HLA_Shared']["fc_len"]
        self.p_dropout = config['HLA_Shared']["p_dropout"]
        w_size = config['HLA_Shared']["w_size"]
        self.linear_input = ((((input_size[1] - conv1_kernel_size) // w_size + 1) // max_pool_size - conv2_kernel_size) // w_size + 1) // 2		# Số đầu vào của lớp fully connected
        self.relu = nn.ReLU().to(self.device)
        self.pool1 = nn.MaxPool1d(2, stride=max_pool_size).to(self.device)
        self.pool2 = nn.MaxPool1d(2, stride=max_pool_size).to(self.device)
        self.conv1 = nn.Conv2d(1, conv1_num_filter, kernel_size=(2, conv1_kernel_size), stride=w_size).to(self.device)		# 2 lớp tích chập
        self.conv2 = nn.Conv1d(conv1_num_filter, conv2_num_filter, kernel_size=conv2_kernel_size, stride=w_size).to(self.device)
        self.bn1 = nn.BatchNorm2d(conv1_num_filter).to(self.device)
        self.bn2 = nn.BatchNorm1d(conv2_num_filter).to(self.device)
        self.fc = nn.Linear(conv2_num_filter * self.linear_input, fc_len).to(self.device)
        self.fc_len = fc_len		# Lớp tuyến tính cuối cùng
        self.fc_bn = nn.BatchNorm1d(fc_len).to(self.device)
        self.HLA_layers = [PrivatedNet(name, fc_len, output_size) 
                           for name, output_size in outputs_size]
        
    def forward(self, x):
        x = x.reshape(-1, 1, self.input_size[0], self.input_size[1])		# Chuyển đầu vào thành dạng 3D numpy array
        out = F.relu(self.bn1(self.conv1(x)))		
        out = out.reshape(-1, out.size()[1], out.size()[3])		# Chuyển đầu vào thành dạng 3D numpy array
        out = self.pool1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        out = out.view(out.size()[0], -1)		# Chuyển đầu ra về dạng vecto 2 chiều, flatten 2 chiều còn lại 
        out = F.dropout(F.relu(self.fc_bn(self.fc(out))), p=self.p_dropout, training=self.training)
        outs = [HLA.forward(out) for HLA in self.HLA_layers]		# Tính các đầu ra ứng với mỗi mạng HLA con 
        out = torch.cat(outs, dim=1)
        return out
    
    def _train(self):
        self.train()
        for HLA in self.HLA_layers:
            HLA.train()		# Hàm gọi chế độ train, thực hiện huấn luyện các mạng con
    
    def _eval(self):
        self.eval()
        for HLA in self.HLA_layers:
            HLA.eval()		# Hàm gọi chế độ test, mạng nơ ron con không được cập nhật trong chế độ này
    
    def predict(self, x):
        x = torch.FloatTensor(np.array(x)).to(self.device).detach()		# Chuyển đầu ra x về dạng torch tensor
        x = x.reshape(-1, self.input_size[0], self.input_size[1])		# Chuyển đầu ra x về dạng 3D numpy array
        output = self.forward(x)
        return output.cpu().data.numpy().flatten()
 
    def set_loss_function(self, loss):
        if loss == "mse":
            self.loss = nn.MSELoss()		# Hàm loss là tổng bình phương sai lệch
        elif loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss == "bce":
            self.loss = nn.BCELoss()		# Hàm loss là binary cross entropy, với đầu ra 2 lớp
        elif loss == "bce_logits":
            self.loss = nn.BCEWithLogitsLoss()		# Hàm BCE logit sau đầu ra dự báo có thêm sigmoid, giống BCE
        elif loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "smooth_l1":
            self.loss = nn.SmoothL1Loss()		# Hàm L1 loss nhưng có đỉnh được làm trơn, khả vi với mọi điểm
        elif loss == "soft_margin":
            self.loss = nn.SoftMarginLoss()		# Hàm tối ưu logistic loss 2 lớp của mục tiêu và đầu ra dự báo
        else:
            raise ValueError("Loss function not found")
        
    def set_optimizer(self, optimizer, lr):
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr)		# Tối ưu theo gradient descent thuần túy
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adadelta":
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr)		# Phương pháp Adadelta có lr update
        elif optimizer == "adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)		# Phương pháp Adagrad chỉ cập nhật lr ko nhớ
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
            
    def reset_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        self.optimizer.step()
          
    def save_train_losses(self, train_losses):
        self.train_losses = train_losses
        
    def save(self, path='./trainned_model', model_name=None):
        if model_name is None:
            model_name = self.name + '_model'
        if not os.path.exists(path):		# Kiểm tra xem đường dẫn khai báo có tồn tại hay không
            os.makedirs(path)
        torch.save(self.state_dict(), "{}/{}.pt".format(path, model_name))
        print("Model saved")
        
    def load(self, path='./trainned_model', model_name=None):
        if model_name is None:
            model_name = self.name + '_model'
        self.load_state_dict(torch.load("{}/{}.pt".format(path, model_name)))		# Lấy lại model đã lưu trước đó
        print("Model loaded")
        
def main():
    model = SharedNet2D(input_size=(2, 101506), outputs_size=[('HLA_A', 42), ('HLA_B', 69), ('HLA_C', 41),
                                                    ('HLA_DRB1', 64), ('HLA_DQA1', 37), ('HLA_DQB1', 30),
                                                    ('HLA_DPA1', 37), ('HLA_DPB1', 37)])
    summary(model, (2, 101506), device='cpu', batch_size=64)
    for _model in model.HLA_layers:
        summary(_model, (256, ), device='cpu', batch_size=64)
    
if __name__ == "__main__":
    main()
