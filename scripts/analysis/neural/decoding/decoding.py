#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Sep 13 17:14:31 2023

@author: rl05
'''

import sys
import argparse
import os
import os.path as op
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import mne

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
from utils import * 


def main():
    
    data_dir = '/imaging/hauk/rl05/fake_diamond/data'
    preprocessed_data_path = op.join(data_dir, 'preprocessed')
    decoding_output_dir = '/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output'

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=str, required=True, help='Run subject-specific analysis')
    parser.add_argument('-a', '--analysis', type=str, required=True, default=None, help='Specify analysis (see README for more)')
    parser.add_argument('-clas', '--classifier', type=str, required=True, default='logistic', help='Specify classifier: logistic or svc')
    parser.add_argument('-data', '--data_type', type=str, required=True, default='MEEG', help='MEEG or MEG or ROI (source space)')
    parser.add_argument('-win', '--window', type=str, required=True, default='temporal', help='Perform analysis on each time point, sliding, or open windows')
    parser.add_argument('--micro_ave', action='store_true', default=False, help='Perform micro-averaging')
    parser.add_argument('--generalise', action='store_true', default=False, help='Perform temporal generalisation')
    # parser.add_argument('--freq', type='store_true', default=False, help='Decompose data into separate frequency bands')
    parser.add_argument('--roi', type=str, required=False, default=None, help='Perform ROI decoding in this ROI')
    args = parser.parse_args()

    subject = f'sub-{args.subject}'
    print(subject)
    analysis = args.analysis
    print('analysis: ', analysis)
    classifier = args.classifier
    print('classifier: ', classifier)
    data_type = args.data_type
    print('data_type: ', data_type)
    generalise = args.generalise
    print('generalise: ', generalise)
    window = args.window
    print('window: ', window)
    roi = args.roi
    print('roi: ', roi)
    print()

    start_time = time.time()

    print('Getting epochs and metadata.')
    epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
    epochs = mne.read_epochs(op.join(epoch_path, f'{subject}_epo.fif'), preload=False, verbose=True)
    epochs.info['bads'] = [] # bads already interpolated during preprocessing; remove bads info for averaging spatial patterns later
    trial_info = pd.read_csv(op.join(data_dir, 'logs', f'{subject}_logfile.csv'))

    epochs.load_data()
    epochs.filter(l_freq=None, h_freq=4.0)
    
    # get epochs drop log as a mask, then apply it to trial info as epochs.metadata
    epochs_drop_mask = [not bool(epoch) for epoch in epochs.drop_log]
    assert(epochs_drop_mask.count(True) == len(epochs.events)) # sanity
    epochs.metadata = trial_info[epochs_drop_mask]

    if roi:
        stcs_epochs_path = op.join(data_dir, 'stcs_epochs', subject)
        stcs_epochs_fname = op.join(stcs_epochs_path, f'stcs_epochs_{roi}_fixed_100Hz.stc.npy')
        if not op.exists(stcs_epochs_fname):
            pass
            print(subject, ' stcs in ', roi, ' not found.')
        else:
            stcs = list(np.load(stcs_epochs_fname, allow_pickle=True)) # stc of a given roi
            X, y = get_analysis_X_y(stcs, epochs.events, epochs.metadata, analysis_name=analysis, spatial=True, window=window) 
    else:
        X, y = get_analysis_X_y(epochs, epochs.events, epochs.metadata, analysis_name=analysis, spatial=False, window=window, micro_averaging=True)
                
    # 1/0

    # some plotting-related parameters
    if analysis.startswith('concreteness_trainWord'):
        times = np.linspace(0.6, 1.4, X[0].shape[2])
    elif analysis.startswith(('concreteness_general','concreteness_train')):
        times = np.linspace(0.6, 1.4, X[0][0].shape[2])
    elif analysis.startswith('specificity_'):
        times = np.linspace(0.6, 1.4, X.shape[2])
    else:
        times = np.linspace(-0.2, 1.4, X.shape[2])

    if data_type == 'MEEG':
        ch_types = ['eeg', 'mag', 'grad']
    elif data_type == 'MEG':
        ch_types = ['mag', 'grad']


    print(f'\nStarting decoding. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
    if roi:
        roi_infix = f'_{roi}'
    else:
        roi_infix = ''
    if generalise:
        if roi:
            analysis_output_dir = op.join(decoding_output_dir, f'{analysis}/timegen/{classifier}/{data_type}/{window}/{subject}/{roi}')
        else:
            analysis_output_dir = op.join(decoding_output_dir, f'{analysis}/timegen/{classifier}/{data_type}/{window}/{subject}')
        if not op.exists(analysis_output_dir):
            os.makedirs(analysis_output_dir, exist_ok=True)
        if analysis.startswith('concreteness_trainWord'):
            X_train = X[0]
            y_train = y[0]
            X_test = X[1]
            y_test = y[1]                
            time_gen = decode_generalise_train(X_train, y_train, classifier=classifier)
            scores = decode_generalise_test(X_test, y_test, estimator=time_gen)
        elif analysis.startswith(('concreteness_general','concreteness_train')):
            n_splits = 10
            scores_splits = []
            for j in range(n_splits):
                X_train = X[0][j]
                y_train = y[0][j]
                X_test = X[1][j]
                y_test = y[1][j]              
                time_gen = decode_generalise_train(X_train, y_train, classifier=classifier)
                scores = decode_generalise_test(X_test, y_test, estimator=time_gen)
                scores_splits.append(scores)
            scores = np.mean(np.array(scores_splits), axis=0)
        else:
            scores = decode_generalise(X, y, analysis=analysis, classifier=classifier)
            scores = np.mean(scores, axis=0) # average scores across cross-validation splits
        np.save(os.path.join(analysis_output_dir, f'scores_time_gen_{data_type}{roi_infix}.npy'), scores)
    else:
        if roi:
            analysis_output_dir = op.join(decoding_output_dir, f'{analysis}/diagonal/{classifier}/{data_type}/{window}/{subject}/{roi}')
        else:
            analysis_output_dir = op.join(decoding_output_dir, f'{analysis}/diagonal/{classifier}/{data_type}/{window}/{subject}')
        if not op.exists(analysis_output_dir):
            os.makedirs(analysis_output_dir, exist_ok=True)
            
        if analysis.startswith('concreteness_trainWord'):
            X_train = X[0]
            y_train = y[0]
            X_test = X[1]
            y_test = y[1]
            time_decod = decode_diagonal_train(X_train, y_train, classifier=classifier)
            scores, coef = decode_diagonal_test(X_test, y_test, estimator=time_decod, classifier=classifier)
        elif analysis.startswith(('concreteness_general','concreteness_train')):
            n_splits = 10
            scores_splits = []
            coef_splits = []
            for j in range(n_splits):
                X_train = X[0][j]
                y_train = y[0][j]
                X_test = X[1][j]
                y_test = y[1][j]            
                time_decod = decode_diagonal_train(X_train, y_train, classifier=classifier)
                scores, coef = decode_diagonal_test(X_test, y_test, estimator=time_decod, classifier=classifier)
                scores_splits.append(scores)
                coef_splits.append(coef)
            scores = np.mean(np.array(scores_splits), axis=0)
            coef = np.mean(np.array(coef_splits), axis=0)
        else:
            scores, coef = decode_diagonal(X, y, analysis=analysis, classifier=classifier)
            scores = np.mean(scores, axis=0) 

        np.save(os.path.join(analysis_output_dir, f'scores_time_decod_{data_type}{roi_infix}.npy'), scores)
        np.save(os.path.join(analysis_output_dir, f'scores_coef_{data_type}{roi_infix}.npy'), coef)

        fig, _ = plot_scores(times=times, scores=scores, analysis=analysis, subject=subject)
        fig.savefig(op.join(analysis_output_dir, f'fig_time_decod{roi_infix}.png'))
        plt.close(fig)

        if not roi:
            fig_evokeds = plot_patterns(coef, info=epochs.info, tmin=epochs.times[0], times='peaks')
            for fig_evoked, ch_type in zip(fig_evokeds, ch_types):
                fig_evoked.savefig(op.join(analysis_output_dir, f'fig_patterns_{data_type}_{ch_type}.png'))
                plt.close(fig_evoked)
    
    print(f'Finished decoding. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')

if __name__ == '__main__':
    main()