#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:36:18 2025

@author: rl05
"""

import sys
import os
import mne
import mne_rsa
import numpy as np
import pandas as pd
import argparse
from time import time

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config

mne.set_log_level(False)  # Be less verbose

data_dir = '/imaging/hauk/rl05/fake_diamond/data'
preprocessed_data_path = os.path.join(data_dir, 'preprocessed')
fig_dir = '/imaging/hauk/rl05/NocturnalBird/figures/rdm'


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', type=str, required=True, help='Run subject-specific analysis')
parser.add_argument('-m', '--metric', type=str, required=True, help='Specify correlation metric')
# parser.add_argument('-data', '--data_type', type=str, required=True, default='MEEG', help='MEEG or MEG or ROI (source space)')
# parser.add_argument('--roi', type=str, required=False, default=None, help='Perform ROI decoding in this ROI')
args = parser.parse_args()

subject = f'sub-{args.subject}'
print(subject)
metric = args.metric
print(metric)
# data_type = args.data_type
# print('data_type: ', data_type)
print()
    

#%% get sensor rdms for all subjects

# conditions = ['low','mid','high']
# metrics = ['correlation','euclidean']

print(metric)

analysis_output_dir = f'/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/rsa/output/{subject}/confusion_matrix'
if not os.path.exists(analysis_output_dir):
    os.makedirs(analysis_output_dir, exist_ok=True)
rdm_fname = os.path.join(analysis_output_dir, f'rdm_{metric}.csv')
# if os.path.exists(rdm_fname):
#     print(f'RDM file exists for {subject}. Skipping.')
#     continue
# else:
print(f'Computing RDM for {subject}.')

start = time()
# read in epochs
print('Reading in epochs.')
epochs_dir = os.path.join(data_dir,'preprocessed',subject,'epoch')
epochs = mne.read_epochs(os.path.join(epochs_dir, f'{subject}_epo.fif'))
# epochs = epochs.resample(100, n_jobs=-1)
epochs = epochs.pick(picks=['meg','eeg'], exclude='bads', verbose=True) # drop bad channels

# compute noise cov for whitening later
print('Computing noise cov.')
noise_cov = mne.compute_covariance(epochs, method='auto', tmin=-0.5, tmax=-0.3, rank='info')
    
epochs.crop(tmin=.9, tmax=1.1) # get data around N400 window
drop_log_mask = [not bool(e) for e in epochs.drop_log] # check which ones are not dropped

logfile = pd.read_csv(os.path.join(data_dir, 'logs',f'{subject}_logfile.csv'))
logfile = logfile[drop_log_mask]
set_nrs = np.sort(logfile['set_nr'].unique()) # get set numbers for this subject
print('Number of sets for this subject: ', len(set_nrs))
# initialize array for getting rdms for each set of items
# rdm_set = np.zeros((len(set_nrs),3)) # 3 is number of condition
df_rdm = pd.DataFrame(index=set_nrs-1).astype('float64')
# df_rdm['set_nr'] = set_nrs
df_rdm[["low_mid", "low_high", "mid_high"]] = None

assert len(epochs) == len(logfile)
epochs.metadata = logfile[['item_nr','set_nr','word1','word2','specificity','experiment']]
mapping = dict(low=0, mid=1, high=2) # for sorting epochs later, to keep track of order in rdm

for i, set_nr in enumerate(set_nrs):
    print('Processing set ', set_nr)
    # print('Set ', set_nr)
    
    # get epochs in a given set
    epochs_target_set = epochs[f"set_nr == {set_nr} and experiment == 'specificity'"].copy()
    
    if len(epochs_target_set) == 0:
        continue
    
    # sort the epochs
    epochs_target_set.metadata['sort_condition'] = epochs_target_set.metadata['specificity'].map(mapping) # turn specificity labels into int
    sort_order = np.argsort(epochs_target_set.metadata['sort_condition']) # get idx for sorting
    epochs_target_set = epochs_target_set[sort_order]
    
    # compute whitener
    print('Whitening data.')
    W, _ = mne.cov.compute_whitener(noise_cov, info=epochs.info, rank='info')
    whitened_data = [W.dot(e) for e in epochs_target_set]
    epochs_target_set._data = np.array(whitened_data)
    
    # compute rdm 
    rdm = mne_rsa.compute_rdm(epochs_target_set, metric=metric)
    
    # check which column to add
    # print('Number of epochs: ', len(epochs_target_set))
    if len(epochs_target_set) == 3:
        df_rdm.loc[i] = rdm.tolist()
    elif len(epochs_target_set) == 2:
        colname = '_'.join(epochs_target_set.metadata['specificity'].tolist())
        # print('Column name: ', colname)
        df_rdm.at[i, colname] = rdm
    else: 
        continue
    
    del rdm
    
df_rdm.to_csv(rdm_fname, index=True)

end = time()
print(f'Done. Time elapsed: {(end-start)/60:.2f} mins.')