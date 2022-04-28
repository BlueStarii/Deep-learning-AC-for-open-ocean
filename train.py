# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:00:52 2021
Requirements:
    PyTorch
    numpy
@author: Jilin Men
"""
import numpy as np
import h5py
import time
# troch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tansformer import tnet
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from tools import EarlyStopping

class myDataset(Dataset):

    def __init__(self, h5_path) -> None:
        super().__init__()

        file_ = h5py.File(h5_path)
        self.x_data = file_['train'][...]
        self.y_data = file_['label'][...]

    def __getitem__(self, index):

        x = self.x_data[index]
        y = self.y_data[index]

        x_mean = np.array([0.04393683,  0.03867437,  0.0325633 ,  0.02249004,  0.02027589,\
                            0.01954756,  0.01208853,  0.01204478,  0.01023882,  0.0092172 ,\
                           -0.00298028,  0.02425979, -0.00978712],'float32')
        x_std = np.array([0.0151591 , 0.01172025, 0.00866078, 0.0081172 , 0.00786252,\
                           0.00772548, 0.00638847, 0.00635459, 0.00579812, 0.00531557,\
                           0.70777417, 0.70828152, 0.70649664],'float32')
        y_mean = np.array([0.00779992, 0.00640577, 0.00509876, 0.00246815, 0.00192338,\
                           0.00165096, 0.00020034, 0.00022317],'float32')
        y_std = np.array([4.39608054e-03, 3.01673863e-03, 1.52137749e-03, 5.82994590e-04,\
                          5.00798373e-04, 4.39324529e-04, 6.75092934e-05, 7.35228105e-05], 'float32')
        x = (x-x_mean)/x_std
        y = (y-y_mean)/y_std
        return x[..., np.newaxis].astype(np.float32).transpose([1, 0]), y[..., np.newaxis].astype(np.float32).transpose([1, 0])

    def __len__(self):

        return self.x_data.shape[0]

def train(train_loader,model,batch_size,opt,device='cuda'):
    model.train()
    
    loss_func = nn.MSELoss()
    total_loss = 0.
    size = len(train_loader.dataset)
    batch = 0
    
    for x,y in train_loader:
        
        x = x.to(device)
        y = y.to(device)
        model.to(device)
        predict = model(x)
        loss = loss_func(predict, y)
        opt.zero_grad()
        loss.backward()

        opt.step()
        total_loss += loss.item()
        log_interval = 100
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            # elapsed = time.time() - start_time
            predict = predict.cpu() 
            y = y.cpu()
            print('| epoch: %d | %d / %d batches | lr:%f | loss: %.4f | predict: %.4f | label:%.4f'%(epoch,
                  batch,int(size/batch_size),opt.param_groups[0]['lr'], cur_loss,
                  predict.detach().numpy().squeeze()[0,0], y.detach().numpy().squeeze()[0,0])) 
            total_loss = 0
            # start_time = time.time()
        batch += 1
    
    return model,cur_loss

def evaluate(model,val_loader,device='cuda'):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    r2_total = 0.
    apd_all = 0.
    loss_func =  nn.MSELoss()
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device)
            # y = y.to(device)
            predict = model(x)
            predict = predict.cpu()
            
            y_label = y.detach().numpy().squeeze()
            pre = predict.detach().numpy().squeeze()
            r2 = r2_score(y_label, pre)
            apd = np.mean((pre-y_label)/y_label)
            loss = loss_func(predict, y)
            total_loss += loss.item()
            r2_total += r2
            apd_all += apd
    return total_loss / (len(val_loader) - 1) ,r2_total/(len(val_loader) - 1),apd_all/(len(val_loader) - 1)
    
if __name__ == "__main__":
    path = r'trainset_CHL.h5'
    ds = myDataset(path)
    rate = 0.7 # ratio of training data
    batch_size = 10000
    epoch = 1000
    data_len = len(ds)
    
    patience = 10
    early_stopping = EarlyStopping(patience,  verbose=True)
    
    indices = torch.randperm(data_len).tolist()
    index = int(data_len * rate)
    train_ds = torch.utils.data.Subset(ds, indices[:index])
    val_ds = torch.utils.data.Subset(ds, indices[index:])
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                                       num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=batch_size,
                                    num_workers=0, drop_last=True, pin_memory=True)
    model = tnet()

    opt = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.99)
    # opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, 
                                                            patience=8, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    for epoch in range(1,epoch+1):
        epoch_start_time = time.time()
        
        model,cur_loss = train(train_loader,model,batch_size,opt)
        val_loss,r2,apd = evaluate(model, val_loader)
        scheduler.step(cur_loss)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.6f} | R2 {:5.2f}| APD {:5.2f} '
        #       .format(epoch, (time.time() - epoch_start_time),val_loss, r2,apd))
        # print('-' * 89)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
             print("Early stopping")

             break        
