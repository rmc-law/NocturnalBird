#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:55:24 2024

@author: rl05
"""

import sys
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
import config 
import helper
from plot_decoding import *


def demo_locatable_axes_easy(ax, ticks, im):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.1)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    cb = plt.colorbar(im, cax=ax_cb, ticks=ticks)
    cb.outline.set_visible(False)
    cb.set_label('AUC', labelpad=-20, rotation=270)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)

to_plot = 'specificity_word'
classifier = 'logistic'
data_type = 'MEEG'
# window = 'single'
window = 'sliding'

subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
figures_dir = op.join(config.project_repo, f'figures/decoding/{to_plot}/{data_type}/{window}')
if not op.exists(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)


if to_plot == 'denotation+concreteness':
    analyses = ['denotation','concreteness']
elif to_plot == 'composition':
    analyses = ['composition']
elif to_plot == 'specificity':
    analyses = ['specificity']
elif to_plot == 'specificity_word':
    analyses = ['specificity_word']
elif to_plot == 'concreteness_xcond':
    analyses = ['concreteness_trainWord_testSub','concreteness_trainWord_testPri']
elif to_plot == 'concreteness_xcond_general':
    analyses = ['concreteness_general_testSub','concreteness_general_testPri']
elif to_plot == 'concreteness_xcond_full':
    analyses = ['concreteness_trainSub_testSub','concreteness_trainSub_testPri',
                'concreteness_trainPri_testSub','concreteness_trainPri_testPri']
  

scores_group = []
for analysis in analyses:
    scores = read_decoding_scores(subjects, analysis, classifier, data_type, window=window, timegen=True)
    scores_group.append(scores)
    del scores

if to_plot == 'denotation+concreteness':
    sfreq = int(scores_group[0].shape[1] / 1.9)
else:
    sfreq = int(scores_group[0].shape[1] / 0.8)
print('sfreq:',sfreq)

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

#%% draw figure
    
if to_plot in ['denotation+concreteness']:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3.85, 6.), sharex=True, sharey=True)
    gs = GridSpec(2, 1, height_ratios=[1,1])
elif to_plot in ['concreteness_xcond','concreteness_xcond_general']:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8., 4.), sharex=True, sharey=True)
    gs = GridSpec(1, 2, width_ratios=[1,1])
elif to_plot == 'composition':
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5., 5.))
elif to_plot == 'concreteness_xcond_full':
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8., 6.), sharex=True, sharey=True)
    gs = GridSpec(2, 2, height_ratios=[1,1])
elif to_plot == 'specificity_word':
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5., 5.))

vmaxs = []

for i, analysis in enumerate(analyses):
        
    if to_plot == 'composition':    
        axis = axes
    if to_plot == 'specificity_word':
        axis = axes
    else:
        axis = plt.subplot(gs[i])

    
    # plot temporal generalisation matrix
    group_avg = np.array(scores_group[i]).mean(axis=0)
    vmax = round(np.max(group_avg), 2)
    vmaxs.append(vmax)
    if (to_plot == 'denotation+concreteness') or (to_plot == 'composition'):
        extent = np.array([-0.2,1.4,-0.2,1.4])
    else:
        extent = np.array([0.,0.8,0.,0.8])
    vmax = np.max(vmaxs)
    im = axis.imshow(group_avg, interpolation=None, origin='lower',
                     cmap=color_scheme[analysis],
                     extent=extent, # times
                      vmin = 0.5, 
                      vmax = vmax
                     )
    
    # some plotting params
    axis.plot(axis.get_xlim(),  axis.get_ylim(), color='lightgrey', ls='--', lw=1) # diagonal
    # add word onsets
    if (to_plot == 'denotation+concreteness') or (to_plot == 'composition'):
        word_onsets = [0., 0.6]
        for onset in word_onsets:
            axis.axhline(onset, color='lightgrey', linewidth=1, linestyle='--')
            axis.axvline(onset, color='lightgrey', linewidth=1, linestyle='--')
    # add labels
    axis.tick_params(axis='both', direction='in')
    if (to_plot == 'denotation+concreteness'):
        axis.set_ylabel('Train Time (s)')
        if i == 1:
            axis.set_xlabel('Test Time (s)') 
    elif (to_plot == 'concreteness_xcond_full'):
        if i in [0,2]:
            axis.set_ylabel('Train Time (s)') 
        if i in [0,1]:
            axis.tick_params(axis='x',labelbottom=False)
        if i in [1,3]:
            axis.tick_params(axis='y',labelleft=False)
        if i in [2,3]:
            axis.set_xlabel('Test Time (s)') 

    # add colorbar
    demo_locatable_axes_easy(axis, [0.5,vmax], im)
    
    # # plot cluster extent
    if (to_plot == 'denotation+concreteness') or (to_plot == 'composition'):
        times = np.linspace(-0.2, 1.4, group_avg.shape[0])
    else:
        times = np.linspace(0.0, 0.8, group_avg.shape[0])
    X, Y = np.meshgrid(times, times)
    
    # for t_threshold in ['1.5','2']:
        # print(t_threshold)
    good_clusters, cluster_pv = permutation_tests(scores_group[i], timegen=True, against_chance=True, t_threshold=None)
    print(f'subplot {i+1}')

    for cluster, pval in zip(good_clusters, cluster_pv):
        # axis.contour(X, Y, cluster, colors=[color])
        if pval < 0.05:
            axis.contour(X, Y, cluster, colors=['black'], linewidths=1)
        else:
            axis.contour(X, Y, cluster, colors=['grey'], linewidths=1)

    
    # # add decomposed testing time courses
    # if to_plot == 'denotation+concreteness':
    #     train_times_idx_to_plot = {'concreteness': [145, 150, 155, 160, 165],
    #                                'denotation': [80, 85, 90, 95, 100]}
    #     for axis_key, train_time_idx in zip(list(axd.keys())[1:], train_times_idx_to_plot[analysis]):
    #         axis = axd[axis_key]
    #         axis.plot(times, group_avg[train_time_idx])
    #         axis.axhline(0.5, color='lightgrey', alpha=0.7, linestyle='--', lw=1, zorder=-20)
    #         axis.set_xlim(-0.2,1.4)
    #         axis.set_ylim(0.48, +0.1)
            
    #     plt.tight_layout()
    #     # plt.savefig(op.join(figures_dir, f'fig_time_gen_group_{classifier}_{data_type}.png'))


# if to_plot == 'composition':
plt.tight_layout()
plt.savefig(op.join(figures_dir, f'fig_time_gen_group_{classifier}_{data_type}_{window}_{sfreq}Hz.png'))
plt.close()