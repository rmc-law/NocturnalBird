#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:01:09 2024

@author: rl05
"""

import os.path as op
import numpy as np
import yaml
from itertools import compress, combinations
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

import mne
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)

decoding_dir = '/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding'
# read in analysis config
with open(op.join(decoding_dir, 'analysis_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)


trigger_scheme = dict(
    low=71,
    mid=81,
    high=91,
    concrete_subsective=111,
    concrete_privative=121,
    concrete_baseline=131,
    abstract_subsective=211,
    abstract_privative=221,
    abstract_baseline=231
)

def get_contrasts(config, analysis_name):
    for analysis in config['analyses']:
        if analysis['name'] == analysis_name:
            target_contrasts = [contrast['name'] for contrast in analysis['contrasts']]
            conditions_to_merge = [contrast['conditions'] for contrast in analysis['contrasts']]
            return target_contrasts, conditions_to_merge

def map_event_ids(trigger_scheme, conditions):
    event_ids = [trigger_scheme[condition] for condition in conditions if condition in trigger_scheme]
    return event_ids

def merge_contrast_events(events, target_contrasts, conditions_to_merge):
    for i in range(len(target_contrasts)):    
        events = mne.merge_events(events, map_event_ids(trigger_scheme, conditions_to_merge[i]), i)
    return events

def sliding_window_average(X, window_size, step, sfreq):
    """
    Applies sliding window averaging to the input data along its last axis (time axis), calculating the average within each window.

    Parameters:
        data (numpy.ndarray): The input data with shape (..., time_samples).
        window_size_ms (float): The size of the sliding window in milliseconds.
        step_size_ms (float): The step size for sliding the window in milliseconds.
        sampling_freq (int): The sampling frequency of the input data in Hz.

    Returns:
        numpy.ndarray: A 3D array containing the averaged values within each sliding window.
    """
    window_size = int(window_size * sfreq / 1000) # convert from ms to samples
    step = int(step * sfreq / 1000) # convert from ms to samples
    num_samples = X.shape[-1] # last axis is time
    num_windows = int((num_samples - window_size) / step) + 1
    output = np.zeros((X.shape[0], X.shape[1], num_windows))
    
    for i in range(num_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        output[..., i] = np.mean(X[..., start_idx:end_idx], axis=-1)
    
    return output

def micro_average(X, Y, nb_ave, analysis):
    # classes = np.unique(Y)
    if analysis == 'specificity_word':
        classes = np.array([0,2],dtype='int32')
    else:
        classes = np.array([0,1],dtype='int32')
    _X, _Y = [], [] # Final X and Y
    print('number of total trials: ',len(X))
    for clas in classes: # loop over classes
        _x, _y = X[Y==clas], Y[Y==clas] # single class X and Y
        print('number of trials in condition ', clas, 'is', len(_x))
        if _x.shape[0] % 2 == 0: # check if number of trials in a condition is even
            for i in tqdm(range(0, _x.shape[0], nb_ave)): 
                comb_x = np.mean(_x[i:i+nb_ave], axis=0) # average pairs of trials in each condition
                _X.append(comb_x)
                _Y.append(clas)
        else: # If odd, loop through the array in steps of 2 for all but the last row
            for i in tqdm(range(0, _x.shape[0] - 1, nb_ave)): 
                comb_x = np.mean(_x[i:i+nb_ave], axis=0) # Average the current pair of rows
                _X.append(comb_x) 
                _Y.append(clas)
            last_row = _x[-1]
            _X.append(last_row)
            _Y.append(clas)
    print('number of total new virtual trials: ', len(_X))
    return np.asarray(_X), np.asarray(_Y)


def get_analysis_X_y(trials, events, metadata, analysis_name='', spatial=False, window='single', micro_averaging=False):
    # input data and events both contain all trials
    
    if spatial:
        X = np.stack([stc._data for stc in trials]) # stc data for all trials
        if analysis_name.startswith('denotation_cross_condition') or analysis_name.startswith('concreteness_'):
            if X[0].shape[1] == 160:
                X = X[:,:,80:] # only analyse second word for this analysis
    else:
        if not trials.preload:
            trials.load_data()
        if analysis_name.startswith('denotation_cross_condition') or analysis_name.startswith('concreteness_') or analysis_name.startswith('specificity_'):
            trials.crop(tmin=0.6, tmax=1.4)
        if trials.info['sfreq'] != 100: 
            trials.resample(100)
        X = trials.get_data()
        
    if window == 'sliding': # implement sliding window analysis
        if analysis_name == 'specificity_word':
            X = sliding_window_average(X, window_size=25, step=10, sfreq=100)
        else:
            X = sliding_window_average(X, window_size=50, step=10, sfreq=100)
        
    target_contrasts, conditions_to_merge = get_contrasts(config, analysis_name)
    events = merge_contrast_events(events, target_contrasts, conditions_to_merge)        
    y = events[:,2]
    
    if analysis_name in ['specificity','specificity_word']:
        experiment = 'specificity'
    else:
        experiment = 'compose'
    mask_experiment = (metadata.experiment==experiment).values 
    metadata = metadata[metadata.experiment==experiment]
    X = np.array(list(compress(X, mask_experiment)))
    y = np.array(list(compress(y, mask_experiment)))
    assert len(X) == len(y)

    # X_microaveraged = np.zeros((X.shape[0],X.shape[1],X.shape[2]))
    # if microaveraging:
    #     if analysis_name == 'composition':
    #         for set_nr in metadata.set_nr.values(): # get stimulus set number
    #             mask_set = ((metadata.set_nr == set_nr) & (metadata.experiment == 'compose') & (metadata.concreteness == 'concrete')).values # make mask
    #             X = np.array(list(compress(X, mask_experiment))) # then average neural responses within a set 

    if analysis_name == 'composition':
        mask_remove_privative = [not t for t in (metadata.denotation=='privative').values] # define composition as subsective>baseline
        X = X[mask_remove_privative]
        y = y[mask_remove_privative]     
    elif analysis_name == 'denotation':
        mask_remove_baseline = [not t for t in (metadata.denotation=='baseline').values] # remove baseline (single word) trials
        X = X[mask_remove_baseline]
        y = y[mask_remove_baseline]
    elif analysis_name.startswith('concreteness_trainWord'):
        mask_get_word = (metadata.denotation=='baseline').values # get single words only for training
        X_train = X[mask_get_word]
        y_train = y[mask_get_word]
        if analysis_name.endswith('testSub'):
            mask_get_subsective = (metadata.denotation=='subsective').values # get subsective trials
            X_test = X[mask_get_subsective]
            y_test = y[mask_get_subsective]
        elif analysis_name.endswith('testPri'):
            mask_get_privative = (metadata.denotation=='privative').values # get private trials
            X_test = X[mask_get_privative]
            y_test = y[mask_get_privative]
        X = (X_train, X_test)
        y = (y_train, y_test)
    elif analysis_name.startswith('concreteness_general'):
        n_splits = 10
        cv = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=42)
        # get train-test splits for subsective trials
        mask_get_subsective = (metadata.denotation=='subsective').values # get subsective trials
        X_sub = X[mask_get_subsective]
        y_sub = y[mask_get_subsective]
        X_sub_train, X_sub_test = [], []
        y_sub_train, y_sub_test = [], []
        for train_index, test_index in cv.split(X_sub, y_sub):
            X_sub_train.append(X_sub[train_index])
            X_sub_test.append(X_sub[test_index])
            y_sub_train.append(y_sub[train_index])
            y_sub_test.append(y_sub[test_index])
        # get train-test splits for privative trials
        mask_get_privative = (metadata.denotation=='privative').values # get private trials
        X_pri = X[mask_get_privative]
        y_pri = y[mask_get_privative]
        X_pri_train, X_pri_test = [], []
        y_pri_train, y_pri_test = [], []
        for train_index, test_index in cv.split(X_pri, y_pri):
            X_pri_train.append(X_pri[train_index])
            X_pri_test.append(X_pri[test_index])
            y_pri_train.append(y_pri[train_index])
            y_pri_test.append(y_pri[test_index])
        X_train, X_test = [], []
        y_train, y_test = [], []
        for i in range(n_splits):
            print('split', i+1)
            # training a general decoder, which has both subsective and privative training trials in it
            X_train.append(np.concatenate((X_sub_train[i], X_pri_train[i]), axis=0)) 
            y_train.append(np.concatenate((y_sub_train[i], y_pri_train[i]), axis=0))
            # X_train.append(X_sub_train[i])
            # X_train.append(X_pri_train[i]) 
            # y_train.append(y_sub_train[i])
            # y_train.append(y_pri_train[i])
            # testing heldout subsective and privative separately
            if analysis_name.endswith('testSub'):
                X_test.append(X_sub_test[i])
                y_test.append(y_sub_test[i])
            if analysis_name.endswith('testPri'):
                X_test.append(X_pri_test[i])
                y_test.append(y_pri_test[i])
        X = (X_train, X_test)
        y = (y_train, y_test)
    elif analysis_name.endswith(('trainSub_testSub','trainSub_testPri','trainPri_testSub','trainPri_testPri')):
        n_splits = 10
        cv = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=42)
        # get train-test splits for subsective trials
        mask_get_subsective = (metadata.denotation=='subsective').values # get subsective trials
        X_sub = X[mask_get_subsective]
        y_sub = y[mask_get_subsective]
        X_sub_train, X_sub_test = [], []
        y_sub_train, y_sub_test = [], []
        for train_index, test_index in cv.split(X_sub, y_sub):
            X_sub_train.append(X_sub[train_index])
            X_sub_test.append(X_sub[test_index])
            y_sub_train.append(y_sub[train_index])
            y_sub_test.append(y_sub[test_index])
        # get train-test splits for privative trials
        mask_get_privative = (metadata.denotation=='privative').values # get private trials
        X_pri = X[mask_get_privative]
        y_pri = y[mask_get_privative]
        X_pri_train, X_pri_test = [], []
        y_pri_train, y_pri_test = [], []
        for train_index, test_index in cv.split(X_pri, y_pri):
            X_pri_train.append(X_pri[train_index])
            X_pri_test.append(X_pri[test_index])
            y_pri_train.append(y_pri[train_index])
            y_pri_test.append(y_pri[test_index])
        X_train, X_test = [], []
        y_train, y_test = [], []
        if analysis_name.startswith('concreteness_trainSub'):
            for i in range(n_splits):
                print('split', i+1)
                X_train.append(X_sub_train[i])
                y_train.append(y_sub_train[i])
                if analysis_name.endswith('testSub'):
                    X_test.append(X_sub_test[i])
                    y_test.append(y_sub_test[i])
                elif analysis_name.endswith('testPri'):
                    X_test.append(X_pri_test[i])
                    y_test.append(y_pri_test[i])    
        elif analysis_name.startswith('concreteness_trainPri'):
            for i in range(n_splits):
                print('split', i+1)
                X_train.append(X_pri_train[i])
                y_train.append(y_pri_train[i])
                if analysis_name.endswith('testSub'):
                    X_test.append(X_sub_test[i])
                    y_test.append(y_sub_test[i])
                elif analysis_name.endswith('testPri'):
                    X_test.append(X_pri_test[i])
                    y_test.append(y_pri_test[i])        
        X = (X_train, X_test)
        y = (y_train, y_test)
    elif analysis_name == 'specificity_word':
        mask_remove_mid = [not t for t in (metadata.specificity=='mid').values]
        X = X[mask_remove_mid]
        y = y[mask_remove_mid]

    if micro_averaging:
        print('Performing micro-averaging.')
        if analysis_name.startswith('concreteness_trainWord'):
            nb_ave = 2 # do a small amount of averaging
            _X, _Y = [], []
            for i, _ in enumerate(['train','test']):
                _x, _y = micro_average(X[i], y[i], nb_ave=nb_ave, analysis=analysis_name)
                _X.append(_x)
                _Y.append(_y)
            X = _X
            y = _Y
        if analysis_name.startswith('concreteness_general'):
            nb_ave = 2 # do a small amount of averaging
            n_splits = 10
            _X_train, _Y_train = [], []
            _X_test, _Y_test = [], []
            for j in range(n_splits): # micro-averaging within train splits
                print('split', i+1)
                _x, _y = micro_average(X[0][j], y[0][j], nb_ave=nb_ave, analysis=analysis_name) # X[0] is the train data
                _X_train.append(_x)
                _Y_train.append(_y)
            for j in range(n_splits): # micro-averaging within test splits
                print('split', i+1)
                _x, _y = micro_average(X[1][j], y[1][j], nb_ave=nb_ave, analysis=analysis_name) # X[1] is the train data
                _X_test.append(_x)
                _Y_test.append(_y)
            X = (_X_train, _X_test)
            y = (_Y_train, _Y_test)
        else:
            if analysis_name == 'concreteness':
                nb_ave = 3 # average over 3 denotation levels
            elif analysis_name == 'denotation':
                nb_ave = 2 # average over 2 denotation levels
            elif analysis_name == 'specificity_word':
                nb_ave = 3 # average over 2 denotation levels
            X, y = micro_average(X, y, nb_ave=nb_ave, analysis=analysis_name) # average together two trials to get better SNR

    return X, y


# =============================================================================
# Decoding functions
# =============================================================================

def choose_pipelines(classifier='', analysis=''):
    if classifier == 'logistic':
        if analysis.startswith('specificity'):
            clf = LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', multi_class='auto', max_iter=10000)
        else:
            clf = LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', multi_class='auto', max_iter=10000)
        clf = make_pipeline(RobustScaler(), LinearModel(clf))
    elif classifier == 'svc':
        clf = make_pipeline(RobustScaler(), LinearModel(SVC(kernel='linear')))
    elif classifier == 'naive_bayes':
        clf = make_pipeline(RobustScaler(), GaussianNB())
    return clf

def choose_scorer(analysis=''):
    if analysis == 'specificity':
        scorer = 'roc_auc_ovo_weighted' # three class problem
    else:
        scorer = 'roc_auc'
    return scorer

def decode_diagonal(X, y, analysis='', classifier=''):
    clf = choose_pipelines(classifier=classifier, analysis=analysis)
    scorer = choose_scorer(analysis=analysis)
    time_decod = SlidingEstimator(clf, scoring=scorer, verbose=True)
    cv = StratifiedKFold(shuffle=True, n_splits=10, random_state=42) 
    scores = cross_val_multiscore(time_decod, X, y, cv=cv) 
    time_decod.fit(X, y) # retrieve spatial patterns and spatial filters for interpretability
    coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
    return scores, coef

def decode_generalise(X, y, analysis='', classifier=''):
    clf = choose_pipelines(classifier=classifier)
    scorer = choose_scorer(analysis=analysis)
    time_gen = GeneralizingEstimator(clf, scoring=scorer, verbose=True)
    cv = StratifiedKFold(shuffle=True, n_splits=10, random_state=42) 
    scores = cross_val_multiscore(time_gen, X, y, cv=cv)
    return scores

# cross condition decoding: train on condition A, test on condition B
def decode_diagonal_train(X_train, y_train, analysis='', classifier=''):
    clf = choose_pipelines(classifier=classifier)
    scorer = choose_scorer(analysis=analysis)
    estimator = SlidingEstimator(clf, scoring=scorer, verbose=True)
    estimator.fit(X_train, y_train)
    return estimator

def decode_diagonal_test(X_test, y_test, estimator=None, classifier=''):
    scores = estimator.score(X_test, y_test)
    if classifier == 'naive_bayes':
        return scores
    else:
        coef = get_coef(estimator, 'patterns_', inverse_transform=True)
        return scores, coef

def decode_generalise_train(X_train, y_train, analysis='', classifier=''):
    clf = choose_pipelines(classifier=classifier)
    scorer = choose_scorer(analysis=analysis)
    estimator = GeneralizingEstimator(clf, scoring=scorer, verbose=True)
    estimator.fit(X_train, y_train)
    return estimator

def decode_generalise_test(X_test, y_test, estimator=None):
    scores = estimator.score(X_test, y_test)
    return scores



# =============================================================================
# Plotting functions
# =============================================================================

def plot_scores(times=None, scores=None, analysis='', subject=''):
    print('Plotting decoding scores.')
    fig, ax = plt.subplots()
    ax.plot(times, scores, label='score')
    ax.axhline(0.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC') 
    ax.legend()
    # ax.axvline(0.0, color='k', linestyle='-')
    ax.set_title(f'Decoding {analysis} in {subject}')
    plt.tight_layout()
    return fig, ax

def plot_patterns(coef=None, info=None, tmin=0., times='peaks'):
    print('Plotting spatial patterns.')
    evoked = mne.EvokedArray(coef, info, tmin=tmin)
    joint_kwargs = dict(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
    fig_evokeds = evoked.plot_joint(times=times, title='patterns', show=False, **joint_kwargs)
    return fig_evokeds