#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:38:47 2024

@author: rl05
"""

import sys
import os
import os.path as op
import numpy as np
import pandas as pd
from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from statsmodels.stats.multitest import multipletests

from mne import read_epochs, combine_evoked, EvokedArray, set_log_level
from mne.viz import plot_compare_evokeds

from scipy.stats import shapiro, ttest_1samp, wilcoxon, kstest, norm

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 

set_log_level(verbose='WARNING')

subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids if subject_id not in ['16']]
print(f'subjects (n={len(subjects)}): ', subjects)
data_dir = op.join(config.project_repo, 'data')
figs_dir = '/imaging/hauk/rl05/NocturnalBird/figures'
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir
stc_path = op.join(data_dir, 'stcs')
analysis = 'specificity'


def calculate_sem(data):
    """
    Calculate the upper and lower confidence margins for 1 SEM from the mean.
    
    Parameters:
        data (numpy.ndarray): Input array of shape (n_observations, n_times).
    
    Returns:
        numpy.ndarray: A 2D array of shape (2, n_times) where:
                       - The first row is the lower margin (mean - SEM)
                       - The second row is the upper margin (mean + SEM)
    """
    # Calculate the mean along the first axis (n_observations)
    mean = np.mean(data, axis=0)
    
    # Calculate the standard error of the mean (SEM)
    sem = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])  # ddof=1 for unbiased estimator
    
    # Calculate the lower and upper margins
    lower_margin = mean - sem
    upper_margin = mean + sem
    
    # Combine the margins into a single array of shape (2, n_times)
    return np.vstack([lower_margin, upper_margin])


# Define parameters
picks = ['EEG033','EEG034','EEG035','EEG044','EEG045',
         'EEG046','EEG055','EEG056','EEG057']
time_windows = {
    'N400_stim1': (0.35, 0.5),
    'N400_stim2': (0.95, 1.1),
    'P600_stim2': (1.2, 1.4)
}
# contrasts = {
#     'high_minus_low': (1, -1, 0),   # weights for [high, mid, low]
#     'mid_minus_low':  (0,  1, -1),
#     'high_minus_mid': (1, -1,  0),  # fix: this should be high - mid
# }



EXCLUDED_WORDS = {'cream', 'horse', 'river', 'stick', 'snacks',
                  'ointment', 'zebra', 'stream', 'icicle', 'crisps'}

logs_dir = op.join(data_dir, 'logs')


# 1. Load all evokeds in one pass per subject
evokeds = {cond: [] for cond in ['high', 'mid', 'low']}
start = time()
# for subject in subjects:
#     print(subject)
#     epochs = read_epochs(
#         op.join(data_dir, 'preprocessed', subject, 'epoch', f'{subject}_epo.fif'),
#         verbose=False, preload=True
#     ).pick(picks)  # pick channels once at load time
#     for cond in ['high', 'mid', 'low']:
#         evokeds[cond].append(epochs[cond].average())
# stop = time()
# print(f'Time took to load data: {(stop-start)/60:.2f}')
for subject in subjects:
    print(subject)
    subject_id = subject.replace('sub-', '')

    # Load full logfile
    log_fname = f'/imaging/hauk/rl05/fake_diamond/data/stcs_epochs/{subject}/epochs_matched_logfile.csv'
    log_df    = pd.read_csv(log_fname).reset_index(drop=True)

    # Load epochs
    epochs = read_epochs(
        op.join(data_dir, 'preprocessed', subject, 'epoch', f'{subject}_epo.fif'),
        verbose=False, preload=True
    ).pick(picks)

    # Sanity check: full logfile should match full epochs
    assert len(log_df) == len(epochs), (
        f"{subject}: logfile has {len(log_df)} rows "
        f"but epochs has {len(epochs)} trials — mismatch!"
    )

    # Attach logfile as metadata
    epochs.metadata = log_df

    # Subset to specificity trials only, then exclude non-hyponymous words
    epochs_spec  = epochs[epochs.metadata['experiment'] == 'specificity']
    epochs_clean = epochs_spec[~epochs_spec.metadata['word2'].isin(EXCLUDED_WORDS)]

    n_removed = len(epochs_spec) - len(epochs_clean)
    print(f"  {subject}: {len(epochs_spec)} specificity trials → "
          f"{len(epochs_clean)} after excluding {n_removed} trials "
          f"({epochs_spec.metadata.loc[epochs_spec.metadata['word2'].isin(EXCLUDED_WORDS), 'word2'].unique().tolist()})")

    for cond in ['high', 'mid', 'low']:
        evokeds[cond].append(epochs_clean[cond].average())

stop = time()
print(f'Time to load: {(stop-start)/60:.2f} min')

# 2. Compute difference waves once
diff_waves = {}
for i_sub in range(len(subjects)):
    diff_waves.setdefault('high_minus_low', []).append(
        combine_evoked([evokeds['high'][i_sub], evokeds['low'][i_sub]], weights=[1, -1]))
    diff_waves.setdefault('mid_minus_low', []).append(
        combine_evoked([evokeds['mid'][i_sub], evokeds['low'][i_sub]], weights=[1, -1]))
    diff_waves.setdefault('mid_minus_high', []).append(
        combine_evoked([evokeds['mid'][i_sub], evokeds['high'][i_sub]], weights=[1, -1]))

# 3. Define contrasts — now referencing diff_waves directly
contrasts = {
    'mid_minus_low':  (diff_waves['mid_minus_low'],  'less'),
    'high_minus_low': (diff_waves['high_minus_low'], 'less'),
    'mid_minus_high': (diff_waves['mid_minus_high'], 'less'),
}


def extract_window_mean(diff_group, picks, tmin, tmax):
    """
    For a list of per-subject difference evokeds, extract a single scalar
    per subject: mean amplitude averaged across specified channels and time window.
    Returns array of shape (n_subjects,).
    """
    return np.array([
        ev.copy().pick(picks).crop(tmin=tmin, tmax=tmax).data.mean(axis=(0, 1))
        for ev in diff_group
    ])

# 3. Run tests
# Collect all results first, then apply FDR correction across all tests
all_results = []

for window_name, (tmin, tmax) in time_windows.items():
    for contrast_name, (diff_group, direction) in contrasts.items():
        
        data = extract_window_mean(diff_group, picks, tmin, tmax)
        t_stat, p_uncorrected = ttest_1samp(data, popmean=0, alternative=direction)
        
        all_results.append({
            'window':       window_name,
            'contrast':     contrast_name,
            'n':            len(data),
            'mean':         data.mean(),
            'sem':          data.std(ddof=1) / np.sqrt(len(data)),
            't':            t_stat,
            'p_uncorrected': p_uncorrected,
            'direction':    direction,
        })

# FDR correction
# results_df = pd.DataFrame(all_results)
# reject, p_fdr, _, _ = multipletests(results_df['p_uncorrected'], method='fdr_bh')
# results_df['p_fdr']   = p_fdr
# results_df['reject']  = reject
results_df_corrected = []

for window, group in results_df.groupby('window'):
    group = group.copy()
    reject, p_fdr, _, _ = multipletests(
        group['p_uncorrected'], method='fdr_bh')
    group['p_fdr']  = p_fdr
    group['reject'] = reject
    results_df_corrected.append(group)

results_df = pd.concat(results_df_corrected).sort_index()

# Report
print("\n=== N400 / Late ERP Results ===\n")
for _, row in results_df.iterrows():
    sig = '✓' if row['reject'] else '✗'
    print(f"[{sig}] {row['window']} | {row['contrast']}")
    print(f"    M = {row['mean']*1e6:.4f} µV (SEM = {row['sem']*1e6:.4f} µV), "
        f"t({int(row['n'])-1}) = {row['t']:.2f}, "
        f"p(uncorr) = {row['p_uncorrected']:.3f}, "
        f"p(FDR) = {row['p_fdr']:.3f}")
    print()
results_df.to_csv('/imaging/hauk/rl05/NocturnalBird/results/neural/erp/erp_ttest_results_robust.csv', index=False)
print("Results saved.")