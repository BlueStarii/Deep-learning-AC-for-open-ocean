The code files were used to achieve atmospheric correction for open ocean water from Rayleigh-corrected products. The inputs are Rayleigh-corrected reflectance at 8 visible and 2 NIR bands, and sola, solz and relaz. The outputs are remote sensing reflectance at 8 visible bands.

create_database.py: create training and testing data from Level-2 products
train.py: build and train the atmospheric correction model
prediction.py: predict on testing data


REQURIEMENTS:
numpy
h5py
glob
torch
