# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:17:12 2021
predict Rrs using Rrc and geometries
@author: Administrator
"""
# import os
import numpy as np
import h5py
# import time
# import math

# troch
import torch
import torch.nn as nn
# import torchvision
from torch.utils.data import Dataset
from tansformer import tnet
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from tools import plot_scatter
import timeit
# from tools import EarlyStopping
# 
def quantile_loss(preds, target, quantiles=[1,1,1,1,1,1,1,1]):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

class myDataset(Dataset):

    def __init__(self, h5_path) -> None:
        super().__init__()

        file_ = h5py.File(h5_path)
        self.x_data = file_['train'][...]
        self.y_data = file_['label'][...]

    
    def __getitem__(self, index):

        # 这个地方可以做想做的预处理~
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
def getLoss(index=0):
    '''
    这个我是参考的这里：https://www.jiqizhixin.com/articles/2018-06-21-3
    '''

    loss_list = [
        nn.MSELoss(),
        nn.SmoothL1Loss(),
        torch.cosh,
        quantile_loss # 这个没测试~
    ]
    if index > 4:
        return loss_list[0]
    else:
        return loss_list[index]
def evaluate(model,val_loader,device='cuda'):
    model.eval() # Turn on the evaluation mode   
    total_label = np.zeros([1,8])
    total_predict = np.zeros([1,8])
    i = 0
    with torch.no_grad():
        for x,y in val_loader:
            
            x = x.to(device)
            y = y.to(device)
            model.to(device)
            predict = model(x)
            # if i %10 == 0 and i != 0:
            #     predict = predict.cpu()
            #     print(str(i+1),'/',str(int(len(val_loader.dataset)/batch_size)))
            total_label = np.concatenate((total_label,y.cpu().detach().numpy().squeeze()), axis=0)
            total_predict = np.concatenate((total_predict,predict.cpu().detach().numpy().squeeze()),axis=0)
            i += 1
    total_label = np.delete(total_label,0,0)
    total_predict = np.delete(total_predict,0,0)
    #反标准化
    # minmax 归一化
    # x_min = np.array([0.000050,0.003899,0.006055,0.004266,0.003491,0.003299,0.001014,0.001039,-1,-1,-1],'float32')
    # x_max = np.array([0.252976,0.207742,0.161734,0.134892,0.132385,0.133128,0.109215,0.110088,1,1,1],'float32')
    # y_min = np.array([0.000098,0.000180,0.000418,0.000712,0.000802,0.000404,0.000004,0.000008],'float32')
    # y_max = np.array([0.060074,0.041702,0.039102,0.022248,0.018130,0.015778,0.000998,0.001660],'float32')
    # total_label = (y_max*0.5-y_min)*total_label+y_min
    # total_predict = (y_max*0.5-y_min)*total_predict+y_min
    # 标准化
    y_mean = np.array([0.00779992, 0.00640577, 0.00509876, 0.00246815, 0.00192338,\
                           0.00165096, 0.00020034, 0.00022317],'float32')
    y_std = np.array([4.39608054e-03, 3.01673863e-03, 1.52137749e-03, 5.82994590e-04,\
                          5.00798373e-04, 4.39324529e-04, 6.75092934e-05, 7.35228105e-05], 'float32')
        
    total_label = total_label*y_std+y_mean
    total_predict = total_predict*y_std+y_mean

    return total_label, total_predict

if __name__ == "__main__":
    starttime = timeit.default_timer()
    path = r'testset_CHL.h5'
    ds = myDataset(path)
    rate = 1 # 训练数据占总数据多少~
    batch_size = 10000
    data_len = len(ds)
    indices = torch.randperm(data_len).tolist()
    index = int(data_len * rate)
    val_ds = torch.utils.data.Subset(ds, indices[:index])
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size,
                                       num_workers=0, drop_last=False, pin_memory=True)

    # model = tnet()
    # model = Net(11,8)
    model = torch.load('best_model_thisstudy.pt')	
    label, predict = evaluate(model, val_loader)
    stats = np.empty((8,6))
    band_name = ['412nm','443nm','488nm','531nm','547nm','555nm','667nm','678nm']
    for band in range(len(band_name)):
        xx = predict[:,band]
        yy = label[:,band]
        ZX = (yy-np.mean(yy))/np.std(yy)
        ZY = (xx-np.mean(xx))/np.std(xx)
        R = np.sum(ZX*ZY)/(len(yy))
        APD = np.mean(abs(xx-yy)/yy)
        RMSE = np.sqrt(np.mean(np.square(xx-yy)))
        unc_RPD = np.std((xx-yy)/yy)
        unc_BIAS = np.std(xx-yy)
        
        stats[band,0] = len(predict)
        stats[band,1] = R
        stats[band,2] = APD
        stats[band,3] = RMSE
        stats[band,4] = unc_RPD
        stats[band,5] = unc_BIAS
        plot_scatter(predict[:,band]*100,label[:,band]*100,band_name[band])
    endtime = timeit.default_timer()
    print('time spending: ', endtime-starttime)
    f = h5py.File('validation_stats.h5','w')
    f.create_dataset('stats',data=stats)
    f.close()
    print('done!')
        
    
    