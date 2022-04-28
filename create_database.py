# -*- coding: utf-8 -*-
'''
please note that any spectrum with Rrs or Rrc <=0 was removed
'''
import numpy as np
import h5py as h5
import glob

from L2wei_QA import QAscores_6Bands
from L2_flags import L3_mask

files = glob.glob(r"./*L2_LAC_OC.nc")
hq_dataset = np.empty([1,21], 'float32')
n = 0

for file in files:
    try:
        n += 1
        data = h5.File(file,"r")
        longitude = data["/navigation_data/longitude"]
        latitude = data["/navigation_data/latitude"]
        geod = ["Rrs_412","Rrs_443","Rrs_488","Rrs_531","Rrs_547","Rrs_555","Rrs_667","Rrs_678",\
                 "rhos_412","rhos_443","rhos_488","rhos_531","rhos_547","rhos_555","rhos_667",\
                "rhos_678","rhos_748","rhos_869","sola","solz","relaz"]
        value = np.empty((longitude.shape[0]*longitude.shape[1], len(geod)),'float32')
        
        for i in range(len(geod)):
                    dataset_band = data["/geophysical_data/" + geod[i]]
                    value_band = dataset_band[:, :] * 1.
                    # value_band[value_band == -32767.] = np.nan
                    value[:,i] = value_band.flatten()
                    try:
                        gain = dataset_band.attrs["scale_factor"][0]
                        offset = dataset_band.attrs["add_offset"][0]
                    except:
                        gain = 1
                        offset = 0
                    value[:,i] = value[:,i]*gain + offset

        flags = [0,1,3,4,5,6,8,9,10,11,12,14,15,16,19,20,21,22,24,25]
        l2flags = data["/geophysical_data/l2_flags"]
        value_masked = L3_mask(flags, l2flags)
        val_index = np.argwhere(value_masked.flatten()==1)      
        L3_data = value[val_index.squeeze(),:]

        # only save QA score=1
        if len(L3_data) !=0:  
            test_lambda = np.array([412,443,488,531,555,667])
            test_Rrs = L3_data[:,:8]
            test_Rrs = np.delete(test_Rrs,[4,7],axis=1)
            maxCos, cos, clusterID, totScore = QAscores_6Bands(test_Rrs, test_lambda)

            hq_rrs_index = np.argwhere(totScore==1) 
            hq_data = L3_data[hq_rrs_index.squeeze(),:]
            hq_dataset = np.concatenate((hq_dataset,hq_data),axis=0)
            print('total data:',len(hq_dataset))

        train_data = hq_dataset[:,8:]
        train_label = hq_dataset[:,:8]

        train_dataset = train_data
        train_dataset[:,-3:] = np.cos(train_data[:,-3:])
        
        file_name = "train_data.h5"
        f = h5.File(file_name, "w")
        f.create_dataset('train', data=train_dataset)
        f.create_dataset('label', data=train_label)
        f.close()
    except:
        print('error')

