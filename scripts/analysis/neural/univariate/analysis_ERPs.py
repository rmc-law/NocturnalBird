#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:38:47 2024

@author: rl05
"""

import os
import os.path as op
import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mne import read_epochs, combine_evoked, EvokedArray, set_log_level
from mne.viz import plot_compare_evokeds

from scipy.stats import shapiro, ttest_1samp, wilcoxon, kstest, norm

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

#%% set figure style 
FONT = 'Arial'
FONT_SIZE = 15
LINEWIDTH = 1.5
EDGE_COLOR = 'grey'
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.transparent': False,
    'axes.labelsize': FONT_SIZE,
    'axes.edgecolor': EDGE_COLOR,
    'axes.linewidth': LINEWIDTH,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'xtick.color': EDGE_COLOR,
    'ytick.color': EDGE_COLOR,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.major.width': LINEWIDTH,
    'ytick.major.width': LINEWIDTH,
    'lines.linewidth' : 3
})

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

# plot with error
fig_erp = plot_compare_evokeds(evokeds, 
                               picks=picks, 
                               combine='mean', 
                               colors=['#ffb14e','#ea5f94','#9d02d7'],
                               ci=calculate_sem,
                               show_sensors=True,
                               title='', legend='upper left')
fig_erp[0].set_size_inches(7.5, 3.5)

axis = fig_erp[0].axes[0]
rect_height = 0.2
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
axis.axvline(x=0.6, color='grey', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig(op.join(figs_dir, 'univariate/erp', 'fig_ERP_group_withError.png'))
plt.close()

# plot without error
fig_erp = plot_compare_evokeds(evokeds, 
                               picks=picks, 
                               combine='mean', 
                               colors=['#ffb14e','#ea5f94','#9d02d7'],
                               ci=None,
                               show_sensors=True,
                               title='', legend='upper left')
fig_erp[0].set_size_inches(7.5, 3.5)
axis = fig_erp[0].axes[0]
rect_height = 0.2
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
axis.axvline(x=0.6, color='grey', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig(op.join(figs_dir, 'univariate/erp', 'fig_ERP_group_withoutError.png'))
plt.close()

#%% plot ERP difference waves

picks = ['EEG033','EEG034','EEG035','EEG044','EEG045','EEG046','EEG055','EEG056','EEG057'] # cente around CPz


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
fig_erp_diff = plot_compare_evokeds(difference, picks=picks, combine="mean", 
                                    ci=calculate_sem, 
                                    show_sensors=True,
                                    colors=['#355f8d', '#22a884', '#bddf26'],
                                    title='' , legend='upper left')
fig_erp_diff[0].set_size_inches(7.5, 3.5)


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
    add_sig_markers(fig_erp_diff[0].axes[0], marker_config)


# add trial structure
axis = fig_erp_diff[0].axes[0]
rect_height = 0.2
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

plt.tight_layout()
plt.savefig(op.join(figs_dir, 'univariate/erp', 'fig_ERP_diff_group.png'))
plt.close()


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



# #%% save N400 data for statistical analysis in R

# # get N400: 300-500 ms window averaged data from each condition 
# N400_contrasts = []
# ERP_diff_conditions = [ERP_diff_group_hm, ERP_diff_group_ml, ERP_diff_group_hl]
# # ERP_diff_conditions = [ERP_diff_group_mh, ERP_diff_group_ml]

# for ERP_diff_group in ERP_diff_conditions: # for each condition
#     # crop data to N400 window, select electrodes, then average across both axes
#     N400_contrasts.extend(np.array([ev.pick(picks).crop(tmin=0.85,tmax=1.1)._data.mean(axis=(0,1)) for ev in ERP_diff_group]))

# # create pd dataframe for statsmodel
# data = pd.DataFrame(dict(N400_amplitude=N400_contrasts,
#                          subject=np.tile(subjects,3),
#                          specificity_contrast=np.repeat(['high-mid','mid-low','high-low'],35)))
# data.to_csv('/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/univariate/N400_data.csv')
# # model = AnovaRM(data, 'N400_amplitude', 'subject', within=['specificity_contrast'])
# # res = model.fit()
# # print(res)

# #%% MNE's spatiotemporal cluster test

# # test difference between N400 effects
# N400_contrasts = []
# ERP_diff_conditions = [ERP_diff_group_hl, ERP_diff_group_ml]

# # test whether there is an N400 effect
# # N400_contrasts = []
# N400_mid_minus_low = np.array([ev._data for ev in ERP_diff_group_ml])
# X = N400_mid_minus_low.transpose((0,2,1))

# for ERP_diff_group in ERP_diff_conditions: # for each condition
#     # crop data to N400 window, select electrodes, then average across both axes
#     N400_contrasts.append(np.array([ev._data for ev in ERP_diff_group]).transpose((0,2,1)))

# # Calculate adjacency matrix between sensors from their locations
# adjacency, _ = find_ch_adjacency(epochs.info, "eeg")
# tfce = dict(start=0.4, step=0.4)  # ideally start and step would be smaller

# print("Clustering.")
# t_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(
#     # N400_contrasts,
#     X, 
#     # tfce,
#     n_permutations=1000,
#     tail=1,
#     # stat_fun=ttest_1samp_no_p,
#     adjacency=adjacency,
#     n_jobs=-1,
#     seed=42
#     )

# # Now select the clusters that are sig. at p < 0.05 (note that this value
# # is multiple-comparisons corrected).
# # good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
# # significant_points = cluster_p_values.reshape(t_obs.shape).T < 0.05
# # print(str(significant_points.sum()) + " points selected by TFCE ...")

# #%% plot clusters

# p_accept = 0.05
# good_cluster_inds = np.where(cluster_p_values < p_accept)[0]
# print(good_cluster_inds)

# # configure variables for visualization
# # colors = dict(low='#ffb14e',mid='#ea5f94',high='#9d02d7')
# # linestyles = {"L": "-", "R": "--"}
# colors = ['#ffb14e']

# # organize data for plotting
# # event_id = dict(low=71, mid=81, high=91)
# # evokeds = {cond: epochs[cond].average() for cond in event_id}
# evokeds = epoch.copy()
# evokeds._data = N400_mid_minus_low.mean(axis=0)

# # loop over clusters
# for i_clu, clu_idx in enumerate(good_cluster_inds):
#     # unpack cluster information, get unique indices
#     time_inds, space_inds = np.squeeze(clusters[clu_idx])
#     ch_inds = np.unique(space_inds)
#     time_inds = np.unique(time_inds)

#     # get topography for F stat
#     f_map = t_obs[time_inds, ...].mean(axis=0)

#     # get signals at the sensors contributing to the cluster
#     sig_times = epochs.times[time_inds]

#     # create spatial mask
#     mask = np.zeros((f_map.shape[0], 1), dtype=bool)
#     mask[ch_inds, :] = True

#     # initialize figure
#     fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")

#     # plot average test statistic and mark significant sensors
#     f_evoked = EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
#     f_evoked.plot_topomap(
#         times=0,
#         mask=mask,
#         axes=ax_topo,
#         cmap="Reds",
#         vlim=(np.min, np.max),
#         show=False,
#         colorbar=False,
#         mask_params=dict(markersize=10),
#     )
#     image = ax_topo.images[0]

#     # remove the title that would otherwise say "0.000 s"
#     ax_topo.set_title("")

#     # create additional axes (for ERF and colorbar)
#     divider = make_axes_locatable(ax_topo)

#     # add axes for colorbar
#     ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(image, cax=ax_colorbar)
#     ax_topo.set_xlabel(
#         "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
#     )

#     # add new axis for time courses and plot time courses
#     ax_signals = divider.append_axes("right", size="300%", pad=1.2)
#     title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
#     if len(ch_inds) > 1:
#         title += "s (mean)"
#     plot_compare_evokeds(
#         evokeds,
#         title=title,
#         picks=picks,
#         axes=ax_signals,
#         colors=colors,
#         # linestyles=linestyles,
#         # linewidths=2,
#         show=False,
#         split_legend=True,
#         truncate_yaxis="auto",
#     )

#     # plot temporal cluster extent
#     ymin, ymax = ax_signals.get_ylim()
#     ax_signals.fill_betweenx(
#         (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
#     )

# plt.show()



# # # We need an evoked object to plot the image to be masked
# # evoked = combine_evoked(
# #     [N400_contrasts[0], N400_contrasts[1]], weights=[1, -1]
# # )  # calculate difference wave
# # time_unit = dict(time_unit="s")
# # evoked.plot_joint(
# #     title="high", ts_args=time_unit, topomap_args=time_unit
# # )  # show difference wave

# # # Create ROIs by checking channel labels
# # selections = make_1020_channel_selections(high_minus_low.info, midline="12z")

# # # Visualize the results
# # fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
# # axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
# # high_minus_low.plot_image(
# #     axes=axes,
# #     group_by=selections,
# #     colorbar=False,
# #     show=False,
# #     mask=significant_points,
# #     show_names="all",
# #     titles=None,
# #     **time_unit,
# # )
# # plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")

# # plt.show()