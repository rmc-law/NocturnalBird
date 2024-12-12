#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:46:46 2024

@author: rl05

Plot ROI effects and calculate effect size
"""

import os
import os.path as op
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from eelbrain import Dataset, load, Factor, plot
from eelbrain._stats.stats import variability
from mne import read_source_spaces, read_source_estimate

import config 
from config_plotting import *

subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids if subject_id not in ['16']]
print(f'subjects (n={len(subjects)}): ', subjects)
data_dir = op.join(config.project_repo, 'data')
figs_dir = op.join(config.project_repo, 'figures')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir
fsaverage_src_fname = op.join(subjects_dir, 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src_fsaverage = read_source_spaces(fsaverage_src_fname, verbose=False)
stc_path = op.join(data_dir, 'stcs')
analysis = input('analysis? (specificity, specificity_word): ')

# label annot
parc = 'semantics'
# annot = read_labels_from_annot('fsaverage_src', parc=parc, hemi='lh')[:-1] # + read_labels_from_annot('fsaverage_src',parc='ventral_ATL')[:2]
# # del annot[2] # to delete visual sources
# assert len(annot) == 4 # should be four rois

results_dir = '/imaging/hauk/rl05/fake_diamond/results/neural/roi/anova/'
times = np.linspace(0., 0.8, 200)




#%% set figure style 
FONT = 'Arial'
FONT_SIZE = 15
LINEWIDTH = 1
EDGE_COLOR = 'grey'
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.transparent': False,
    'axes.labelsize': FONT_SIZE,
    'axes.edgecolor': EDGE_COLOR,
    'axes.linewidth': LINEWIDTH,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'xtick.color': EDGE_COLOR,
    'ytick.color': EDGE_COLOR,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': LINEWIDTH,
    'ytick.major.width': LINEWIDTH
})

#%% read in dataset using eelbrain

if analysis == 'specificity': 
    conditions = ['low','mid','high']
    roi_to_plot = input('ATL or all: ')
    colors = plt.cm.Oranges([0.4, 0.6, 0.8]).tolist()
    relevant_conditions = conditions
elif analysis == 'specificity_word': 
    conditions = ['low','high']
    roi_to_plot = input('ATL or all: ')
    colors = plt.cm.Oranges([0.4, 0.8]).tolist()
    relevant_conditions = conditions

subjects_list, conditions_list, stcs = [], [], []
for subject in subjects:
    print(f'Reading in stc {subject}.')
    for condition in conditions:
        stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_MEEG-lh.stc')
        stc = read_source_estimate(stc_fname, subject='fsaverage_src')
        stc = stc.crop(tmin=0.6, tmax=1.4)
        stc.tmin = 0.
        stcs.append(stc)
        subjects_list.append(subject)
        conditions_list.append(condition)
        del stc

ds = Dataset()
concreteness = [condition.split('-')[0] for condition in conditions_list]
if analysis == 'specificity':
    specificity = [condition.split('-')[0] for condition in conditions_list]
elif analysis == 'specificity_word':
    specificity_word = [condition.split('-')[0] for condition in conditions_list]
ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='oct-6', parc=parc) 
ds['subject'] = Factor(subjects_list, random=True)
ds['condition'] = Factor(conditions_list)
ds['condition'].sort_cells(conditions)
if analysis == 'specificity':
    ds['specificity'] = Factor(specificity)
    ds['specificity'].sort_cells(['low','mid','high'])
elif analysis == 'specificity_word':
    ds['specificity_word'] = Factor(specificity_word)
    ds['specificity_word'].sort_cells(['low','high'])
stc_reset = ds['stcs']


#%% plot figures
    
fig_dir = op.join(config.project_repo, 'figures', 'univariate')

if roi_to_plot == 'ATL':
    rois = ['anteriortemporal-lh','anteriortemporal-rh']
    roi_names = ['LATL','RATL']
    n_subplots = len(rois)
    fig_name = op.join(fig_dir, f'{analysis}_timecourse_ATLs.png')
else:
    rois = ['anteriortemporal-lh','anteriortemporal-rh','posteriortemporal-lh','posteriortemporal-rh',
            'inferiorfrontal-lh','inferiorfrontal-rh','temporoparietal-lh','temporoparietal-rh']
    roi_names = ['anterior temporal', 'posterior temporal', 'inferior frontal', 'temporo parietal']
    n_subplots = len(rois)
    fig_name = op.join(fig_dir, f'{analysis}_timecourse_allROIs.png')

if n_subplots == 2:
    fig_height = 2.5
    fig_width = 10 # for two subplots sidebyside
    fig, axes = plt.subplots(1, n_subplots, sharex=True, sharey=True, dpi=300)
    fig.set_size_inches(fig_width, fig_height)
else:
    fig, axes = plt.subplots(int(n_subplots/2), 2, sharex=True, sharey='row', dpi=300)
    fig.set_size_inches(10, 8)
# fig.subplots_adjust(hspace=0.45)

for i_roi, roi in enumerate(rois):
    if roi_to_plot == 'all':
        if i_roi < 2:
            axis = axes[0][i_roi]
        elif i_roi < 4:
            i_roi -= 2
            axis = axes[1][i_roi]
        elif i_roi < 6:
            i_roi -= 4
            axis = axes[2][i_roi]
        elif i_roi < 8:
            i_roi -= 6
            axis = axes[3][i_roi]
    elif roi_to_plot == 'ATL':
        axis = axes[i_roi]
    ds['stcs'] = stc_reset
    stcs = ds['stcs']
    stcs_region = stcs.sub(source = roi)
    time_courses = stcs_region.mean('source')
    ds['stcs'] = time_courses
    output_dir = op.join(results_dir, roi, analysis)

    # plot condition time courses with 1 within-subjects sem
    for relevant_condition, color in zip(relevant_conditions,colors):
        data = ds['stcs'][ds[analysis].isin([relevant_condition])]
        data_group_avg = data.mean('case') # average over subjects
        error = variability(y=data.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
        axis.plot(times, data_group_avg.x, color=color, lw=3)
        axis.fill_between(times, data_group_avg.x-error, data_group_avg.x+error, alpha=0.2, color=color)
        if roi_to_plot == 'all':
            axis.title.set_text(roi)
    axis.set_xlim(0., 0.8)

    xticks = [0., 0.2, 0.4, 0.6, 0.8]
    plt.xticks(xticks)
    axis.set_xlabel('Time (s)')
    if roi_to_plot == 'ATL':
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(True)
    if i_roi == 0:
        axis.set_ylabel('Current (Am)')
        
    # add custom legend
    legend_elements = [Line2D([0], [0], color=color, lw=3) for color in colors]
    if analysis == 'specificity_word': 
        fig.get_axes()[0].legend(legend_elements, ['low', 'high'], loc='upper right')

    # read in permutation test pickle file
    print('unloading pickle: ', roi)
    pickle_fname = op.join(results_dir, f'{roi}/{analysis}/{roi}.pickle')
    with open(pickle_fname, 'rb') as f:
        res = pickle.load(f)
    pmin = 0.1
    mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
    sign_clusters = res.clusters[mask_sign_clusters]

    if sign_clusters.n_cases != None: #check for significant clusters
        for i in range(sign_clusters.n_cases):
            cluster_tstart = sign_clusters[i]['tstart'] - 0.6
            cluster_tstop = sign_clusters[i]['tstop'] - 0.6
            cluster_effect = sign_clusters[i]['effect']
            cluster_pval = sign_clusters[i]['p']
            if analysis == 'specificity_word':
                cluster_effect = 'specificity_word'
            if cluster_effect != analysis:
                continue
            if cluster_pval < 0.05:
                alpha = 0.3
                # color_cluster = color_scheme['significant']
                color_cluster = 'yellow'
            else:
                alpha = 0.2
                # color_cluster = color_scheme['marginal']
                color_cluster = 'grey'
            # axis.axhline(data_group_avg.x.min(), cluster_tstart, cluster_tstop, color='yellow', alpha=alpha, linewidth=3)#, zorder=-50)
            axis.axvspan(cluster_tstart, cluster_tstop, color=color_cluster, zorder=-100, alpha=alpha)

            output_dir = op.join(results_dir, roi, 'replicateB&P')
            if not op.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            roi_activity = time_courses.mean(time=(cluster_tstart,cluster_tstop))
            
            # split conditions
            if analysis == 'specificity':
                effect_split = 'specificity'
                xtick_labels = ['low','mid','high']
                colors_bar = colors
            elif analysis == 'specificity_word':
                effect_split = 'specificity_word'
                xtick_labels = ['low','high']
                colors_bar = colors
                
            def label_bars(bar, *argv):
                for i, arg in enumerate(argv):
                    # height = bar[i].get_height()
                    axis_bar.annotate(arg,
                                  xy=(bar[i].get_x() + bar[i].get_width() / 2, 0),
                                  xytext=(0, 3),  # 3 points vertical offset
                                  textcoords="offset points",
                                  ha='center', va='bottom',
                                  fontsize=FONT_SIZE, color='white',
                                  rotation=90, zorder=100)
    
            fig_bar, axis_bar = plt.subplots(figsize=(3,2.5))
            width = 0.3
            j = 0
            if analysis == 'composition':
                x_positions = [1,1.5,2.25,2.75]
                for c in ['concrete','abstract']:
                    for d_label, d in zip(['word','phrase'],['baseline','subsective']):
                        cond = ds['stcs'][ds['concreteness'].isin([c]) & ds['composition'].isin([d])]
                        error = variability(y=cond.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
                        cond_bar_mean = cond.mean(time=(cluster_tstart,cluster_tstop)).x.mean()
                        cond_bar_err = error[int(cluster_tstart*250):int(cluster_tstop*250)].mean()
                        bar = axis_bar.bar(x_positions[j], cond_bar_mean, width, yerr=cond_bar_err, color=colors_bar[j])
                        label_bars(bar, d_label)
                        j += 1
                axis_bar.set_xticklabels(['concrete', 'abstract'])
                axis_bar.set_xticks([(x_positions[0]+x_positions[1])/2, (x_positions[2]+x_positions[3])/2])
            elif analysis == 'denotation':
                x_positions = [1,1.5,2.25,2.75]
                for c in ['concrete','abstract']:
                    for d_label, d in zip(['subsective','privative'],['subsective','privative']):
                        cond = ds['stcs'][ds['concreteness'].isin([c]) & ds['denotation'].isin([d])]
                        error = variability(y=cond.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
                        cond_bar_mean = cond.mean(time=(cluster_tstart,cluster_tstop)).x.mean()
                        cond_bar_err = error[int(cluster_tstart*250):int(cluster_tstop*250)].mean()
                        bar = axis_bar.bar(x_positions[j], cond_bar_mean, width, yerr=cond_bar_err, color=colors_bar[j])
                        label_bars(bar, d_label)
                        j += 1
                axis_bar.set_xticklabels(['concrete', 'abstract'])
                axis_bar.set_xticks([(x_positions[0]+x_positions[1])/2, (x_positions[2]+x_positions[3])/2])
            elif analysis == 'specificity':
                x_positions = [1,1.5,2,2.5]
                for d_label, d in zip(['low','mid', 'high'],['low','mid','high']):
                    cond = ds['stcs'][ds['specificity'].isin([d])]
                    error = variability(y=cond.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
                    cond_bar_mean = cond.mean(time=(cluster_tstart,cluster_tstop)).x.mean()
                    cond_bar_err = error[int(cluster_tstart*250):int(cluster_tstop*250)].mean()
                    bar = axis_bar.bar(x_positions[j], cond_bar_mean, width, yerr=cond_bar_err, color=colors_bar[j])
                    # label_bars(bar, d_label)
                    j += 1
                axis_bar.set_xticklabels(['low', 'mid', 'high'])
                axis_bar.set_xticks([x_positions[0], x_positions[1], x_positions[2]])
            elif analysis == 'specificity_word':
                x_positions = [1,1.5,2,2.5]
                for d_label, d in zip(['low', 'high'],['low','high']):
                    cond = ds['stcs'][ds['specificity_word'].isin([d])]
                    error = variability(y=cond.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
                    cond_bar_mean = cond.mean(time=(cluster_tstart,cluster_tstop)).x.mean()
                    cond_bar_err = error[int(cluster_tstart*250):int(cluster_tstop*250)].mean()
                    bar = axis_bar.bar(x_positions[j], cond_bar_mean, width, yerr=cond_bar_err, color=colors_bar[j])
                    # label_bars(bar, d_label)
                    j += 1
                axis_bar.set_xticklabels(['low', 'high'])
                axis_bar.set_xticks([x_positions[0], x_positions[1]])
            axis_bar.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
            axis_bar.spines['top'].set_visible(False)
            axis_bar.spines['right'].set_visible(False)
            axis_bar.spines['left'].set_visible(False)
            axis_bar.spines['bottom'].set_visible(True)
            axis_bar.set_ylabel('Current (Am)')
            fig_bar.tight_layout()
            fig_bar.savefig(op.join(fig_dir, f'{cluster_effect}_{roi}_c{i+1}_split-bars.png'))
            plt.close(fig_bar)

plt.tight_layout()
fig.savefig(fig_name)
plt.close()


