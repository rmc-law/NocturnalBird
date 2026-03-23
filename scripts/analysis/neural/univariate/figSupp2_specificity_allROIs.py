#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 2 21:46:46 2025

@author: rl05

Plot Supplementary Figure - specificity effects in all tested ROIs
"""

import os
import os.path as op
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from eelbrain import Dataset, load, Factor
from eelbrain._stats.stats import variability
from mne import read_source_estimate
import pickle

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
from config_plotting import *
import fig_constants
import fig_helpers as fh
import config

# 1. CONFIGURATION
roi          = 'anteriortemporal-lh'
analysis     = 'specificity'
conditions   = ['low','mid','high']
subjects     = [f"sub-{s}" for s in config.subject_ids]

subjects_dir = op.join(config.project_repo, 'data/mri')
os.environ['SUBJECTS_DIR'] = subjects_dir
stc_path = op.join(config.project_repo, 'data/stcs')
results_dir = '/imaging/hauk/rl05/NocturnalBird/results/neural/roi/anova/' # contains pickled permutation results
figures_dir = '/imaging/hauk/rl05/NocturnalBird/figures/paper/'
os.makedirs(figures_dir, exist_ok=True)
colors = [color_scheme[condition] for condition in conditions]
times = np.linspace(0., 0.8, 200)

# 2. LOAD IN DATASET USING EELBRAIN
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
ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='oct-6', parc='semantics') 
ds['subject'] = Factor(subjects_list, random=True)
ds['specificity'] = Factor(conditions_list)
ds['specificity'].sort_cells(conditions)
stc_reset = ds['stcs']


# 3. SET UP FIGURE 
mosaic = [
    ['A','B'],
    ['C','D'],
    ['E','F'],
    ['G','H']]
fig, ax_dict = plt.subplot_mosaic(
    mosaic,  # Specify the layout of subplots using the mosaic parameter
    figsize=(fig_constants.FIG_WIDTH, 6.5),  # Set the size of the figure in inches
    dpi=300,  # Set the resolution of the figure in dots per inch
    constrained_layout=True,  # Enable constrained layout for automatic adjustment
    # sharey='row',
    gridspec_kw={
        'height_ratios': [1,1,1,1], # Set the relative heights of the rows
        'width_ratios': [1,1], # Set the relative widths of the columns
        'wspace': 0.001,
        'hspace': 0.005}
)

rois = ['anteriortemporal-lh', 'anteriortemporal-rh',
        'posteriortemporal-lh', 'posteriortemporal-rh',
        'inferiorfrontal-lh', 'inferiorfrontal-rh',
        'temporoparietal-lh', 'temporoparietal-rh']
subplot_labels = ['A','B','C','D','E','F','G','H']


for roi, subplot_label in zip(rois, subplot_labels):
    ds['stcs'] = stc_reset
    stcs = ds['stcs']
    stcs_region = stcs.sub(source = roi)
    time_courses = stcs_region.mean('source')
    ds['stcs'] = time_courses
    axis = ax_dict[subplot_label]
    for condition, color in zip(conditions,colors):
        data = ds['stcs'][ds[analysis].isin([condition])]
        data_group_avg = data.mean('case') # average over subjects
        error = variability(y=data.x[:,:], x=ds['specificity'], match=ds['subject'], pool=True, spec='sem')
        axis.plot(times, data_group_avg.x, color=color, lw=2, label=condition)
        axis.fill_between(times, data_group_avg.x-error, data_group_avg.x+error, alpha=0.2, color=color)
        axis.title.set_text(roi)
    axis.set_xlim(0., 0.8)
    axis.set_xticks([0., 0.2, 0.4, 0.6, 0.8])
    if subplot_label in ['G', 'H']:
        axis.set_xlabel('Time (s)')
    if subplot_label in ['A']:
        legend_elements = [Line2D([0], [0], color=color, lw=3) for color in colors]
        axis.legend(legend_elements, ['low','mid','high'], loc='upper right', frameon=False)         
        axis.set_ylabel('Dipole moment (Am)')

    # add brain model inset
    imgA = mpimg.imread(f'/imaging/hauk/rl05/NocturnalBird/figures/labels/fig_roi_label_{roi}_low_contrast.png')
    imagebox = OffsetImage(imgA, zoom=0.045)  # adjust zoom as needed
    ab = AnnotationBbox(
        imagebox,
        xy=(0.0, 1.0),             # upper-right in axis coordinates
        xycoords='axes fraction', # interpret xy as relative to axes
        box_alignment=(0., 1.),     # align image top-right
        frameon=False             # no border around image
    )
    axis.add_artist(ab)

    # add cluster extents if any
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
            if cluster_effect != analysis:
                continue
            if cluster_pval < 0.05:
                alpha = 0.3
                color_cluster = 'yellow'
            else:
                alpha = 0.2
                # color_cluster = color_scheme['marginal']
                color_cluster = 'grey'
            x_limits = (0.0, 0.8)
            span_coords = [(cluster_tstart, cluster_tstop)]
            alphas = [0.25]
            fh.add_background_spans(axis, span_coords, x_limits, alphas, color=color_cluster)

fh.label_panels_mosaic(fig, ax_dict, size = 14)

# plt.suptitle('Main effect of composition in anterior temporal lobe', fontweight='bold')

# 9. FINALIZE & SAVE
out_fname = op.join(figures_dir, 'figSupp1_specificity_allROIs.png')
plt.savefig(out_fname, dpi=300)
plt.close()
print(f"Saved combined figure to {out_fname}.")

