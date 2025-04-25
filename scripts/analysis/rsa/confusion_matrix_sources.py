#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:08:43 2025

@author: rl05

Compute confusion matrix
"""

import sys
import os
import mne
import mne_rsa
import numpy as np
import pandas as pd
import argparse
from time import time

from scipy.spatial.distance import squareform
from itertools import accumulate

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config

mne.set_log_level(False)  # Be less verbose

data_dir = '/imaging/hauk/rl05/fake_diamond/data'
preprocessed_data_path = os.path.join(data_dir, 'preprocessed')
fig_dir = '/imaging/hauk/rl05/NocturnalBird/figures/rdm'

subjects = [subject for subject in config.subject_ids if subject not in ['16']]
print(f'subjects (n={len(subjects)}): ', subjects)


# parser = argparse.ArgumentParser()
# parser.add_argument('-s', '--subject', type=str, required=True, help='Run subject-specific analysis')
# parser.add_argument('-m', '--metric', type=str, required=True, help='Specify correlation metric')
# # parser.add_argument('-data', '--data_type', type=str, required=True, default='MEEG', help='MEEG or MEG or ROI (source space)')
# # parser.add_argument('--roi', type=str, required=False, default=None, help='Perform ROI decoding in this ROI')
# args = parser.parse_args()

# subject = f'sub-{args.subject}'
# print(subject)
# metric = args.metric
# print(metric)
# # data_type = args.data_type
# # print('data_type: ', data_type)
# print()

#%% get source rdms for all subjects

# conditions = ['low','mid','high']

# subjects_dir = os.path.join(data_dir, 'mri')
# os.environ['SUBJECTS_DIR'] = subjects_dir
# label_ATL = mne.read_labels_from_annot('fsaverage_src', parc='semantics', hemi='lh')[0]

# fsaverage_src_fname = os.path.join(subjects_dir, 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
# src_fsaverage = mne.read_source_spaces(fsaverage_src_fname, verbose=False)

metrics = ['correlation','euclidean']

for metric in metrics:
    print(metric)
    for subject in subjects:
        
        start = time()

        subject = f'sub-{subject}'
        print(subject)
        analysis_output_dir = f'/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/rsa/output/{subject}/confusion_matrix'
        if not os.path.exists(analysis_output_dir):
            os.makedirs(analysis_output_dir, exist_ok=True)

        rdm_fname = os.path.join(analysis_output_dir, f'rdm_ATL-fixed_{metric}.npy')

        # if os.path.exists(rdm_fname):
        #     print(f'RDM file exists for {subject}. Skipping.')
        #     continue
        # else:
        
        subject_data_dir = os.path.join(data_dir, 'stcs_epochs', subject)
        
        # read in subject stcs
        stc_epochs = np.load(os.path.join(subject_data_dir,'stcs_epochs_anteriortemporal-lh_fixed.stc.npy'), allow_pickle=True) # use signed data for more info
        stc_epochs = [stc.crop(tmin=0.9, tmax=1.1) for stc in stc_epochs] # get only time window of interest
        
        # read in epochs only to get drop log to get logfile
        epochs_dir = os.path.join(data_dir,'preprocessed',subject,'epoch')
        epochs = mne.read_epochs(os.path.join(epochs_dir, f'{subject}_epo.fif'), preload=False)
        drop_log_mask = [not bool(e) for e in epochs.drop_log] #
        logfile = pd.read_csv(os.path.join(data_dir, 'logs',f'{subject}_logfile.csv'))
        logfile = logfile[drop_log_mask].reset_index(names='old_index')
        logfile = logfile.reset_index(names='new_index')
        # logfile = logfile[drop_log_mask]
        
        assert len(stc_epochs) == len(logfile)
        epochs.metadata = logfile
        epochs = epochs["experiment == 'specificity'"].copy()

        set_nrs = np.sort(epochs.metadata['set_nr'].unique()) # get set numbers for this subject
        print('Number of sets for this subject: ', len(set_nrs))
        
        # initialize dataframe
        df_rdm = pd.DataFrame(index=set_nrs-1).astype('float64')
        df_rdm[["low_mid", "low_high", "mid_high"]] = None
        
        mapping = dict(low=0, mid=1, high=2) # for sorting epochs later, to keep track of order in rdm

        for i, set_nr in enumerate(set_nrs):
            print('Processing set ', set_nr)
            
            # again, use epochs class to help subset stc_epochs
            epochs_target_set = epochs[f"set_nr == {set_nr} and experiment == 'specificity'"].copy() # get trials from a set
            if len(epochs_target_set) == 0:
                continue
            epochs_target_set.metadata['sort_condition'] = epochs_target_set.metadata['specificity'].map(mapping) # turn specificity labels into int
            sort_order = np.argsort(epochs_target_set.metadata['sort_condition']) # get idx for sorting            
            epochs_target_set = epochs_target_set[sort_order]

            set_idx = epochs[f"set_nr == {set_nr} and experiment == 'specificity'"].metadata['new_index'].tolist()
            stc_epochs_target_set = [stc_epochs[idx] for idx in set_idx] # get the target trials
            stc_epochs_target_set = [stc_epochs[k] for k in sort_order] # then sort into low-mid-high
            stc_epochs_target_set = [s._data for s in stc_epochs_target_set]

            # compute rdm 
            rdm = mne_rsa.compute_rdm(stc_epochs_target_set, metric=metric)
            
            # check which column to add
            if len(stc_epochs_target_set) == 3:
                df_rdm.loc[i] = rdm.tolist()
            elif len(stc_epochs_target_set) == 2:
                colname = '_'.join(epochs_target_set.metadata['specificity'].tolist())
                # print('Column name: ', colname)
                df_rdm.at[i, colname] = rdm
            else: 
                continue
            
            del rdm
            
        df_rdm.to_csv(rdm_fname, index=True)
        
        end = time()
        print(f'Done. Time elapsed: {(end-start)/60:.2f} mins.')
        del epochs_target_set, stc_epochs_target_set, logfile, epochs, stc_epochs
            
            # # initialize array for getting rdms for each set of items
            # rdm_set = np.zeros((len(set_nrs),3)) # 3 is number of condition
            
            # for i_set, set_nr in enumerate(set_nrs):
            #     set_idx = epochs.metadata[(logfile.set_nr == set_nr) & (logfile.experiment=='specificity')].index.tolist()
            #     if len(set_idx) < 2: # if that set only has 1 item, skip it
            #         continue
            #     # elif any(idx >= len(stc_epochs) for idx in set_idx): # if item out of bounds, skip
            #     #     continue
            #     # get epoch counts
            #     # epoch_counts = np.array([epochs[condition].__len__() for condition in conditions])
                
            #     # compute rdm 
            #     target_stc_epochs = [stc_epochs[i]._data for i in set_idx]
            #     rdm = mne_rsa.compute_rdm(target_stc_epochs, metric=metric)
                # rdm_set[i_set] = rdm
                
            # np.save(rdm_fname, rdm_set)

