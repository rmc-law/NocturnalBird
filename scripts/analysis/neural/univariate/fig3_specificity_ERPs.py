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
import pandas as pd

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

results_df = pd.read_csv('/imaging/hauk/rl05/NocturnalBird/results/neural/erp/erp_ttest_results.csv')

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

# rect_height = 0.15  # 15% of the axis height
# rect_width = 0.2    # 20% of the axis width
# rect_y = -0.25      # Positioned 25% of the axis height below the plot area

# axis = fig_erp[0].axes[0]
axis = ax_dict['A']
rect_height = 0.3
rect_width = 0.3
rect_y = -1.75
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
axis.title.set_text('raw event-related potentials')
xticks = [-0.5, -0.3, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
axis.set_xticks(xticks)
axis.set_xticklabels([f'{t:.1f}' for t in xticks])

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

# (i.e. absolute times, not noun-relative)
WINDOW_COORDS = {
    'N400_stim1': (0.35, 0.500),
    'N400_stim2': (0.95, 1.100),
    'P600_stim2': (1.200, 1.396),
}

CONTRAST_COLOURS = {
    'mid_minus_low':  '#22a884',
    'high_minus_low': '#bddf26',
    'high_minus_mid': '#355f8d'
}

WINDOW_Y_ANCHOR = {
    'N400_stim1': 0.42,
    'N400_stim2': 0.48,
    'P600_stim2': 0.41,
}

# Apply significant markers to the axis
# for marker_config in marker_configs:
#     add_sig_markers_from_results(ax_dict['B'], marker_config)

# RECT_HEIGHT = 0.04   # height of each coloured bar in µV (data units)
# RECT_Y_BASE = 0.42   # y position of the bottom bar — adjust to your axis limits
# RECT_SPACING = 0.06  # gap between the two bars
RECT_HEIGHT  = 0.04
RECT_SPACING = 0.05   # vertical gap between stacked bars
STAR_OFFSET  = 0.02   # gap between top bar and asterisk


def build_marker_configs(results_df, window_coords, contrast_colours,
                          window_y_anchor, rect_height, rect_spacing,
                          star_offset, p_threshold=0.1):
    """
    One single coloured bar per significant contrast per window.
    Colour encodes the contrast. Solid = FDR. Dashed = trend.
    Contrasts within the same window are offset vertically.
    """
    sig = (results_df[results_df['p_fdr'] < p_threshold]
           .sort_values(['window', 'p_uncorrected'])
           .copy())

    configs = []

    for window, group in sig.groupby('window'):
        if window not in window_coords:
            continue

        x_start, x_end = window_coords[window]
        width           = x_end - x_start
        y_anchor_base   = window_y_anchor[window]

        for i_contrast, (_, row) in enumerate(group.iterrows()):
            contrast = row['contrast']
            p_uncorr = row['p_uncorrected']
            p_fdr    = row['p_fdr']

            if contrast not in contrast_colours:
                continue

            # One bar per contrast, offset vertically within window
            y_pos     = y_anchor_base + i_contrast * (rect_height + rect_spacing)
            colour    = contrast_colours[contrast]   # single colour string
            linestyle = '-' if p_fdr < 0.05 else '--'

            position_y = y_pos + rect_height + star_offset

            if p_uncorr < 0.001:   marker = '***'
            elif p_uncorr < 0.01:  marker = '**'
            elif p_uncorr < 0.05:  marker = '*'
            elif p_uncorr < 0.1:   marker = '†'
            else:                  continue

            configs.append({
                'position_x': x_start,
                'position_y': position_y,
                'width':      width,
                'height':     rect_height,
                'color':      colour,     # single colour, not a list
                'y_pos':      y_pos,
                'marker':     marker,
                'linestyle':  linestyle,
            })

    return configs


def add_sig_markers(axis, bar_config):
    linestyle = bar_config['linestyle']
    color     = bar_config['color']
    y_centre  = bar_config['y_pos'] + bar_config['height'] / 2
    x_start   = bar_config['position_x']
    x_end     = bar_config['position_x'] + bar_config['width']

    axis.plot(
        [x_start, x_end],
        [y_centre, y_centre],
        color=color,
        linewidth=1,
        linestyle=linestyle,   # '-' or '--' directly
        alpha=0.9,
        zorder=5,
        solid_capstyle='butt'  # flat ends, no overhang
    )

    # Marker text
    # axis.text((x_start + x_end) / 2, y_centre + 0.05,
    #           bar_config['marker'],
    #           ha='center', va='bottom', fontsize=10, zorder=6)

marker_configs = build_marker_configs(
    results_df       = results_df,
    window_coords    = WINDOW_COORDS,
    contrast_colours = CONTRAST_COLOURS,
    window_y_anchor  = WINDOW_Y_ANCHOR,
    rect_height      = RECT_HEIGHT,
    rect_spacing     = RECT_SPACING,
    star_offset      = STAR_OFFSET,
    # use_fdr          = True,    # uncorrected for visualisation
    p_threshold      = 0.1,
)

# Step 1: check what the CSV contains for the late window
print("=== All late_stim2 rows in CSV ===")
print(results_df[results_df['window'] == 'late_stim2'][
    ['window', 'contrast', 'p_uncorrected', 'p_fdr', 'reject']])

# Step 2: check what survives the p_threshold filter
print("\n=== Rows passing p_uncorrected < 0.05 ===")
print(results_df[results_df['p_uncorrected'] < 0.1][
    ['window', 'contrast', 'p_uncorrected', 'p_fdr', 'reject']])

# Step 3: check what configs were actually built
print("\n=== Built marker_configs ===")
for c in marker_configs:
    print(c)

# Step 4: check your axis y limits BEFORE adding patches
print("\n=== ax_dict['B'] ylim ===")
print(ax_dict['B'].get_ylim())

# Step 5: check your axis x limits
print("\n=== ax_dict['B'] xlim ===")
print(ax_dict['B'].get_xlim())

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
axis.title.set_text('pairwise ERP difference waves')
axis.set_xticks(xticks)
axis.set_xticklabels([f'{t:.1f}' for t in xticks])

fh.label_panels_mosaic(fig, ax_dict, size = 14)


# plt.tight_layout()
out_fname = op.join(figs_dir, 'paper', 'fig3_specificity_ERPs.png')
plt.savefig(out_fname, dpi=300)
plt.close()
print(f"Saved combined figure to {out_fname}.")