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
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mne import read_epochs, combine_evoked, EvokedArray, set_log_level
from mne.viz import plot_compare_evokeds

from scipy.stats import shapiro, ttest_1samp, wilcoxon, kstest, norm

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 
from config_plotting import *
import fig_constants
import fig_helpers as fh


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

#%% read in ERP data

evoked_low_group = []
evoked_mid_group = []
evoked_high_group = []
for i_subject, subject in enumerate(subjects):
    print(subject)
    epochs = read_epochs(op.join(data_dir,'preprocessed',subject,'epoch',f'{subject}_epo.fif'),verbose=False).pick('eeg')
    epochs = epochs[['low','mid','high']]
    evoked_low_group.append(epochs['low'].average())
    evoked_mid_group.append(epochs['mid'].average())
    evoked_high_group.append(epochs['high'].average())


#%% plot ERPs 

evokeds = dict(low=evoked_low_group, mid=evoked_mid_group, high=evoked_high_group)
picks = ['EEG033','EEG034','EEG035','EEG044','EEG045','EEG046','EEG055','EEG056','EEG057'] # cente around CPz


# make fig 
mosaic = [
    ['A'],
    ['B']]
fig, ax_dict = plt.subplot_mosaic(
    mosaic,  # Specify the layout of subplots using the mosaic parameter
    figsize=(fig_constants.FIG_WIDTH*0.75, 4),  # Set the size of the figure in inches
    dpi=300,  # Set the resolution of the figure in dots per inch
    constrained_layout=True,  # Enable constrained layout for automatic adjustment
    # sharey='row',
    gridspec_kw={
        'height_ratios': [1,1], # Set the relative heights of the rows
        'width_ratios': [1], # Set the relative widths of the columns
        'wspace': 0.001,
        'hspace': 0.005}
)

# plot with error
fig_erp = plot_compare_evokeds(
    evokeds, 
    picks=picks, 
    combine='mean', 
    colors=['#ffb14e','#ea5f94','#9d02d7'],
    ci=calculate_sem,
    show_sensors=True,
    title='', legend='upper left',
    truncate_yaxis=False,
    truncate_xaxis=False,
    axes=ax_dict['A'],
    show=False
)
# fig_erp[0].set_size_inches(fig_constants.FIG_WIDTH, 3)

# define box properties for annotating timings

rect_height = 0.15  # 15% of the axis height
rect_width = 0.2    # 20% of the axis width
rect_y = -0.25      # Positioned 25% of the axis height below the plot area

# axis = fig_erp[0].axes[0]
axis = ax_dict['A']
# rect_height = 0.3
# rect_width = 0.3
# rect_y = -1.75
rect_xs = [-0.3, 0., 0.6]
texts = ['+', 'stim 1', 'stim 2']
for rect_x, text in zip(rect_xs,texts):
    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                          linewidth=1., edgecolor='black', facecolor='lightgrey')
    axis.add_patch(rect)
    text_x = rect_x + rect_width / 2
    text_y = rect_y + rect_height / 2
    axis.text(text_x, text_y, text, ha='center', va='center')
axis.axvline(x=0.6, color='black', linestyle='--', linewidth=1)
axis.title.set_text('event-related potentials')
# fig_erp[0].savefig(op.join(figs_dir, 'paper', 'fig1_specificity_ERPs.png'))
# plt.close()

# plot ERP difference waves
# picks = ['EEG033','EEG034','EEG035','EEG044','EEG045','EEG046','EEG055','EEG056','EEG057'] # cente around CPz

ERP_diff_group_mh = [] # mid minus high
ERP_diff_group_ml = [] # mid minus low
ERP_diff_group_hl = [] # high minus low
for i_subject, _ in enumerate(subjects):    
    mid_minus_high = combine_evoked([evoked_mid_group[i_subject], evoked_high_group[i_subject]], weights=[1, -1])
    mid_minus_low = combine_evoked([evoked_mid_group[i_subject], evoked_low_group[i_subject]], weights=[1, -1])
    high_minus_low = combine_evoked([evoked_high_group[i_subject], evoked_low_group[i_subject]], weights=[1, -1])
    ERP_diff_group_mh.append(mid_minus_high)
    ERP_diff_group_ml.append(mid_minus_low)
    ERP_diff_group_hl.append(high_minus_low)
difference = dict(mid_minus_high=ERP_diff_group_mh, mid_minus_low=ERP_diff_group_ml, high_minus_low=ERP_diff_group_hl)
fig_erp_diff = plot_compare_evokeds(
    difference, 
    picks=picks, 
    combine="mean", 
    ci=calculate_sem, 
    show_sensors=True,
    colors=['#355f8d', '#22a884', '#bddf26'],
    title='' , legend='upper left',
    truncate_yaxis=False,
    truncate_xaxis=False,
    axes=ax_dict['B'],
    show=False    
)
# fig_erp_diff[0].set_size_inches(7.5, 3.5)


# add some annotations to the figure
def add_sig_markers(axis, bar_config):
    """
    Adds significant markers (stars and rectangles) to the given axis.

    Parameters:
        axis (matplotlib axis): The axis to which markers will be added.
        bar_config (dict): Configuration for the markers, including positions, colors, and dimensions.
    """
    # Add significance star text marker
    text_x = bar_config['position_x'] + bar_config['width'] / 2
    text_y = bar_config['position_y'] + bar_config['height'] / 2
    axis.text(text_x, text_y, '*', ha='center', va='center')

    # Add colored rectangles
    for rect_y, color in zip(bar_config['rect_ys'], bar_config['colors']):
        rect = patches.Rectangle((bar_config['position_x'], rect_y), bar_config['width'], 0.04,
                                 linewidth=2., edgecolor=None, facecolor=color)
        axis.add_patch(rect)



# Define configurations for significant markers
marker_configs = [
    { # N400 at stimulus position 1
        'position_x': 0.35,
        'position_y': 0.45,
        'width': 0.15,
        'height': 0.1,
        'colors': ['#22a884', '#355f8d'],
        'rect_ys': [0.45, 0.40],
    },
    { # N400 at stimulus position 2
        'position_x': 0.95,
        'position_y': 0.5,
        'width': 0.15,
        'height': 0.1,
        'colors': ['#bddf26', '#22a884'],
        'rect_ys': [0.5, 0.45],
    },
    {  # N600 at stimulus position 2
        'position_x': 1.2,
        'position_y': 0.45,
        'width': 0.196,
        'height': 0.1,
        'colors': ['#22a884', '#355f8d'],
        'rect_ys': [0.45, 0.40],
    }
]

# Apply significant markers to the axis
for marker_config in marker_configs:
    add_sig_markers(ax_dict['B'], marker_config)


# add trial structure
# axis = fig_erp_diff[0].axes[0]
axis = ax_dict['B']
rect_height = 0.3
rect_width = 0.3
rect_y = -1.25
rect_xs = [-0.3, 0., 0.6]
texts = ['+', 'stim 1', 'stim 2']
for rect_x, text in zip(rect_xs,texts):
    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                          linewidth=1., edgecolor='black', facecolor='lightgrey')
    axis.add_patch(rect)
    text_x = rect_x + rect_width / 2
    text_y = rect_y + rect_height / 2
    axis.text(text_x, text_y, text, ha='center', va='center')
axis.axvline(x=0.6, color='black', linestyle='--', linewidth=1)
axis.title.set_text('N400 effects of specificity')

fh.label_panels_mosaic(fig, ax_dict, size = 14)


# plt.tight_layout()
out_fname = op.join(figs_dir, 'paper', 'fig1_specificity_ERPs.png')
plt.savefig(out_fname, dpi=300)
plt.close()
print(f"Saved combined figure to {out_fname}.")







#%% Tests for N400 effects

stimulus_positions = [(0.352,0.5),(0.952,1.1)]

# get N400 effect: subtract baseline (low specificity) from target (mid or high)
contrast_names = ['high_minus_low','mid_minus_low','high_minus_mid']
ERP_diff_conditions = [ERP_diff_group_hl, ERP_diff_group_ml, ERP_diff_group_mh]

for (tmin, tmax) in stimulus_positions:
    print(tmin, tmax)
    N400_contrasts = []
    for ERP_diff_group in ERP_diff_conditions: # for each condition
        data = deepcopy(ERP_diff_group)
        # crop data to N400 window, select electrodes, then average across both axes
        N400_contrasts.append(np.array([ev.pick(picks).crop(tmin=tmin,tmax=tmax)._data.mean(axis=(0,1)) for ev in data]))
    
    # check for normality
    # Perform the Shapiro-Wilk test
    for data, contrast_name in zip(N400_contrasts, contrast_names):
        print()
        print('Contrast: ', contrast_name)
        # stat, p_value = shapiro(data)
        stat, p_value = kstest(data, "norm", args=(np.mean(data), np.std(data)))
    
        # print("Shapiro-Wilk Test Statistic:", stat)
        print("Kolmogorov-Smirnov Test Statistic:", stat)
    
        print("P-value:", p_value)
        
        # Interpret the result
        if p_value < 0.05:
            print("The data is not normally distributed (p < 0.05).")
            print('Using Wilcoxon Signed-Rank Test.')
            stat, p_value = wilcoxon(data, alternative='less')
            
            print("Wilcoxon Statistic:", stat)
            print("P-value:", p_value)
            
            # Interpret results
            if p_value < 0.05:
                print("The median is significantly different from 0 (p < 0.05).")
            else:
                print("The median is not significantly different from 0 (p >= 0.05).")
            
        else:
            print("The data is normally distributed (p >= 0.05).")
            print('Using One-Sample t-Test.')
    
            # Perform a one-sample t-test against 0
            t_stat, p_value = ttest_1samp(data, 0, alternative='less')
            
            print("T-statistic:", t_stat)
            print("P-value:", p_value)
            
            # Interpret results
            if p_value < 0.05:
                print("The mean is significantly different from 0 (p < 0.05).")
            else:
                print("The mean is not significantly different from 0 (p >= 0.05).")
                
                
    with open(op.join(figs_dir,'univariate','erp','ttest_N400_results.txt'), "w") as file:
        for data, contrast_name in zip(N400_contrasts, contrast_names):
            file.write("\n")
            file.write(f"Contrast: {contrast_name}\n")
            
            # Perform Kolmogorov-Smirnov Test
            stat, p_value = kstest(data, "norm", args=(np.mean(data), np.std(data)))
            file.write(f"Kolmogorov-Smirnov Test Statistic: {stat}\n")
            file.write(f"P-value: {p_value}\n")
            
            if p_value < 0.05:
                file.write("The data is not normally distributed (p < 0.05).\n")
                file.write("Using Wilcoxon Signed-Rank Test.\n")
                
                # Perform Wilcoxon test
                stat, p_value = wilcoxon(data, alternative="less")
                file.write(f"Wilcoxon Statistic: {stat}\n")
                file.write(f"P-value: {p_value}\n")
                
                if p_value < 0.05:
                    file.write("The median is significantly different from 0 (p < 0.05).\n")
                else:
                    file.write("The median is not significantly different from 0 (p >= 0.05).\n")
            else:
                file.write("The data is normally distributed (p >= 0.05).\n")
                file.write("Using One-Sample t-Test.\n")
                
                # Perform a one-sample t-test
                t_stat, p_value = ttest_1samp(data, 0, alternative="less")
                file.write(f"T-statistic: {t_stat}\n")
                file.write(f"P-value: {p_value}\n")
                
                if p_value < 0.05:
                    file.write("The mean is significantly different from 0 (p < 0.05).\n")
                else:
                    file.write("The mean is not significantly different from 0 (p >= 0.05).\n")
    
    
    #%% Tests for later effects
    
    # get effect: subtract baseline (low specificity) from target (mid or high)
    N400_contrasts = []
    contrast_names = ['high_minus_low','mid_minus_low','high_minus_mid']
    ERP_diff_conditions = [ERP_diff_group_hl, ERP_diff_group_ml, ERP_diff_group_mh]
    for ERP_diff_group in ERP_diff_conditions: # for each condition
        # crop data to N400 window, select electrodes, then average across both axes
        N400_contrasts.append(np.array([ev.pick(picks).crop(tmin=1.2,tmax=1.396)._data.mean(axis=(0,1)) for ev in ERP_diff_group]))
    
    # check for normality
    # Perform the Shapiro-Wilk test
    for data, contrast_name in zip(N400_contrasts, contrast_names):
        print()
        print('Contrast: ', contrast_name)
        # stat, p_value = shapiro(data)
        stat, p_value = kstest(data, "norm", args=(np.mean(data), np.std(data)))
    
        # print("Shapiro-Wilk Test Statistic:", stat)
        print("Kolmogorov-Smirnov Test Statistic:", stat)
    
        print("P-value:", p_value)
        
        # Interpret the result
        if p_value < 0.05:
            print("The data is not normally distributed (p < 0.05).")
            print('Using Wilcoxon Signed-Rank Test.')
            stat, p_value = wilcoxon(data, alternative='less')
            
            print("Wilcoxon Statistic:", stat)
            print("P-value:", p_value)
            
            # Interpret results
            if p_value < 0.05:
                print("The median is significantly different from 0 (p < 0.05).")
            else:
                print("The median is not significantly different from 0 (p >= 0.05).")
            
        else:
            print("The data is normally distributed (p >= 0.05).")
            print('Using One-Sample t-Test.')
    
            # Perform a one-sample t-test against 0
            t_stat, p_value = ttest_1samp(data, 0, alternative='less')
            
            print("T-statistic:", t_stat)
            print("P-value:", p_value)
            
            # Interpret results
            if p_value < 0.05:
                print("The mean is significantly different from 0 (p < 0.05).")
            else:
                print("The mean is not significantly different from 0 (p >= 0.05).")
                
                
    with open(op.join(figs_dir,'univariate','erp','ttest_N600_results.txt'), "w") as file:
        for data, contrast_name in zip(N400_contrasts, contrast_names):
            file.write("\n")
            file.write(f"Contrast: {contrast_name}\n")
            
            # Perform Kolmogorov-Smirnov Test
            stat, p_value = kstest(data, "norm", args=(np.mean(data), np.std(data)))
            file.write(f"Kolmogorov-Smirnov Test Statistic: {stat}\n")
            file.write(f"P-value: {p_value}\n")
            
            if p_value < 0.05:
                file.write("The data is not normally distributed (p < 0.05).\n")
                file.write("Using Wilcoxon Signed-Rank Test.\n")
                
                # Perform Wilcoxon test
                stat, p_value = wilcoxon(data, alternative="less")
                file.write(f"Wilcoxon Statistic: {stat}\n")
                file.write(f"P-value: {p_value}\n")
                
                if p_value < 0.05:
                    file.write("The median is significantly different from 0 (p < 0.05).\n")
                else:
                    file.write("The median is not significantly different from 0 (p >= 0.05).\n")
            else:
                file.write("The data is normally distributed (p >= 0.05).\n")
                file.write("Using One-Sample t-Test.\n")
                
                # Perform a one-sample t-test
                t_stat, p_value = ttest_1samp(data, 0, alternative="less")
                file.write(f"T-statistic: {t_stat}\n")
                file.write(f"P-value: {p_value}\n")
                
                if p_value < 0.05:
                    file.write("The mean is significantly different from 0 (p < 0.05).\n")
                else:
                    file.write("The mean is not significantly different from 0 (p >= 0.05).\n")

