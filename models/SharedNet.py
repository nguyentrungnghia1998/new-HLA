#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import numpy as np
from torch import optim
from torchsummary.torchsummary import summary

config = json.load(open('models/config.json'))

class PrivatedNet(nn.Module):
    def __init__(self, name, input_size, output_size):
        super(PrivatedNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fc1_len = config[name]["fc1_len"]
        fc2_len = config[name]["fc2_len"]     
        self.fc1 = nn.Linear(input_size, fc1_len).to(self.device)
        self.bn1 = nn.BatchNorm1d(fc1_len).to(self.device)
        self.fc2 = nn.Linear(fc1_len, fc2_len).to(self.device)
        self.fc3 = nn.Linear(fc2_len, output_size).to(self.device)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.fc2(out)
        out = self.fc3(out)
        return torch.softmax(out, dim=1)
        

class SharedNet(nn.Module):
    def __init__(self, input_size, outputs_size):
        super(SharedNet, self).__init__()
        self.name = 'SharedNet'
        self.input_size = input_size
        self.outputs_size = outputs_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conv1_num_filter = config['HLA_Shared']["conv1_num_filter"]
        conv2_num_filter = config['HLA_Shared']["conv2_num_filter"]
        conv1_kernel_size = config['HLA_Shared']["conv1_kernel_size"]
        conv2_kernel_size = config['HLA_Shared']["conv2_kernel_size"]
        fc_len = config['HLA_Shared']["fc_len"]
        linear_input = (((input_size - conv1_kernel_size + 4) // 4) - conv2_kernel_size + 3) // 4
        self.relu = nn.ReLU().to(self.device)
        self.pool = nn.MaxPool1d(2, stride=4).to(self.device)
        self.conv1 = nn.Conv1d(1, conv1_num_filter, kernel_size=conv1_kernel_size).to(self.device).to(self.device)
        self.conv2 = nn.Conv1d(conv1_num_filter, conv2_num_filter, kernel_size=conv2_kernel_size).to(self.device)
        self.bn1 = nn.BatchNorm1d(conv1_num_filter).to(self.device)
        self.bn2 = nn.BatchNorm1d(conv2_num_filter).to(self.device)
        self.fc = nn.Linear(conv2_num_filter * linear_input, fc_len).to(self.device)
        self.HLA_layers = [PrivatedNet(name, fc_len, output_size) 
                           for name, output_size in outputs_size]
        self.fc_len = fc_len
        
    def forward(self, x):
        x = x.reshape(-1, 1, self.input_size)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        outs = [HLA.forward(out) for HLA in self.HLA_layers]
        out = torch.cat(outs, dim=1)
        return out
    
    def _train(self):
        self.train()
        for HLA in self.HLA_layers:
            HLA.train()
    
    def _eval(self):
        self.eval()
        for HLA in self.HLA_layers:
            HLA.eval()
    
    def predict(self, x):
        x = torch.FloatTensor(np.array(x)).to(self.device).detach() 
        x = x.reshape(-1, self.input_size)
        output = self.forward(x)
        return output.cpu().data.numpy().flatten()
 
    def set_loss_function(self, loss):
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss == "bce":
            self.loss = nn.BCELoss()
        elif loss == "bce_logits":
            self.loss = nn.BCEWithLogitsLoss()
        elif loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "smooth_l1":
            self.loss = nn.SmoothL1Loss()
        elif loss == "soft_margin":
            self.loss = nn.SoftMarginLoss()
        else:
            raise ValueError("Loss function not found")
        
    def set_optimizer(self, optimizer, lr):
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adadelta":
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr)
        elif optimizer == "adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
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
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), "{}/{}.pt".format(path, model_name))
        print("Model saved")
        
    def load(self, path='./trainned_model', model_name=None):
        if model_name is None:
            model_name = self.name + '_model'
        self.load_state_dict(torch.load("{}/{}.pt".format(path, model_name)))
        print("Model loaded")
        
def main():
    model = SharedNet(input_size=101506, outputs_size=[('HLA_A', 42), ('HLA_B', 69), ('HLA_C', 41),
                                                    ('HLA_DRB1', 64), ('HLA_DQA1', 37), ('HLA_DQB1', 30),
                                                    ('HLA_DPA1', 37), ('HLA_DPB1', 37)])
    summary(model, (1, 101506), device='cpu', batch_size=1)
    for _model in model.HLA_layers:
        summary(_model, (256, ), device='cpu', batch_size=1)
    
if __name__ == "__main__":
    main()