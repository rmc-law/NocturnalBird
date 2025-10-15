import os
import os.path as op
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from eelbrain import Dataset, load, Factor, concatenate
# from eelbrain._stats.stats import Dispersion
from eelbrain._stats.stats import variability
from mne import read_source_estimate
import pickle

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing') 
from config_plotting import *
import fig_constants
import fig_helpers as fh
import config


# 1. CONFIGURATION
roi         = 'inferiorfrontal'
analysis    = 'specificity'
conditions  = ['low','mid','high']
hemis       = ['lh','rh']
subjects     = [f"sub-{s}" for s in config.subject_ids]

subjects_dir = op.join(config.project_repo, 'data/mri')
os.environ['SUBJECTS_DIR'] = subjects_dir
stc_path = op.join(config.project_repo, 'data/stcs')
results_dir = '/imaging/hauk/rl05/NocturnalBird/results/neural/roi/anova/' # contains pickled permutation results
figures_dir  = op.join('/imaging/hauk/rl05/NocturnalBird/figures/paper/')
os.makedirs(figures_dir, exist_ok=True)
colors = [color_scheme[condition] for condition in conditions]
times = np.linspace(0., 0.8, 200)

# 2. LOAD IN DATASET USING EELBRAIN
subjects_list, conditions_list, hemis_list, regions_list, stcs = [], [], [], [], []
for subject in subjects:
    print(f'Reading in stc {subject}.')
    for hemi in hemis: 
        for condition in conditions:
            roi_name = roi + '-' + hemi
            stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_MEEG-{hemi}.stc')
            stc = read_source_estimate(stc_fname, subject='fsaverage_src')
            stc = stc.crop(tmin=0.6, tmax=1.4)
            # stc = load.mne.stc_ndvar(stc, subject='fsaverage_src', src='oct-6', parc='semantics')
            # stc = stc.sub(source=roi_name).mean('source')
            stc.tmin = 0.
            stcs.append(stc)
            subjects_list.append(subject)
            conditions_list.append(condition)
            hemis_list.append(hemi)
            regions_list.append('inferiorfrontal')
            del stc

ds = Dataset()
specificity = conditions_list
ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='oct-6', parc='semantics') 
# ds['stcs'] = concatenate(stcs, dim='case')
ds['subject'] = Factor(subjects_list, random=True)
ds['specificity'] = Factor(conditions_list)
ds['specificity'].sort_cells(conditions)
ds['hemi'] = Factor(hemis_list)
ds['region'] = Factor(regions_list)
stc_reset = ds['stcs']



# get hemi-specific data
ds_lh = ds.sub("hemi == 'lh'")
ds_rh = ds.sub("hemi == 'rh'")

# average within each ATL
stcs_lh = ds_lh['stcs'].sub(source='inferiorfrontal-lh').mean('source')
stcs_rh = ds_rh['stcs'].sub(source='inferiorfrontal-rh').mean('source')
time_courses_avg_clean = (stcs_lh + stcs_rh) / 2.0

ds_avg = Dataset()
ds_avg['subject'] = ds_lh['subject']
ds_avg['specificity'] = ds_lh['specificity']
ds_avg['stcs'] = time_courses_avg_clean # Assign the correctly averaged data

print(f"Length of final averaged dataset: {len(ds_avg['stcs'])}")
# Expected Output: Length of final averaged dataset: 105

all_errors = variability(
    y=ds_avg['stcs'],  # The complete averaged NDVar
    x=ds_avg[analysis],             # The complete 'specificity' factor
    match=ds_avg['subject'],        # The complete 'subject' factor
    pool=False,
    spec='sem'
)


# 3. LOAD STATISTICAL RESULTS
pickle_fname = op.join(results_dir, 'bilateralIFG/specificity*hemisphere.pickle') # Hypothetical pickle file
try:
    with open(pickle_fname, 'rb') as f:
        res = pickle.load(f)
    pmin = 0.05 # Using pmin to show marginal effects if any
    mask_sign_clusters = (res.clusters['p'] <= pmin) & (res.clusters['effect'] == analysis)
    sign_clusters = res.clusters[mask_sign_clusters]
    # sign_clusters = sign_clusters[0]
    print(f"Loaded {sign_clusters.n_cases} significant/marginal clusters from {pickle_fname}")
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
except FileNotFoundError:
    print(f"Warning: Statistical results file not found at {pickle_fname}. Plotting without stats.")
    sign_clusters = None



# 4. SET UP FIGURE 
mosaic = [
    ['A','B']]
fig, ax_dict = plt.subplot_mosaic(
    mosaic,  # Specify the layout of subplots using the mosaic parameter
    figsize=(4.5, 1.8),  # Set the size of the figure in inches
    dpi=300,  # Set the resolution of the figure in dots per inch
    constrained_layout=True,  # Enable constrained layout for automatic adjustment
    # sharey='row',
    gridspec_kw={
        'height_ratios': [1], # Set the relative heights of the rows
        'width_ratios': [2,0.7], # Set the relative widths of the columns
        'wspace': 0.001,
        'hspace': 0.005}
)


# 5. PANEL A: Time series for bilateral ATL

line_zorders = [99,98,97]
axis = ax_dict['A']
for i, (condition, color, zorder) in enumerate(zip(conditions, colors, line_zorders)):
    # data_for_condition = ds['stcs'][ds[analysis].isin([condition])]
    data_for_condition = ds_avg['stcs'][ds_avg[analysis].isin([condition])] # bilaterally-averaged data for this condition
    data_group_avg = data_for_condition.mean('case') # average over subjects
    # error = variability(
    #     y=data_for_condition.x[:,:], 
    #     x=ds_avg[analysis][ds_avg[analysis].isin([condition])], 
    #     # match=ds['subject'][ds[analysis].isin([condition])],
    #     match=ds['subject'], 
    #     pool=True, 
    #     spec='sem'
    #     )
    # print(error)
    error_for_condition = all_errors[i]
    axis.plot(times, data_group_avg.x, color=color, lw=2.5, label=condition, zorder=zorder)
    axis.fill_between(times, data_group_avg.x-error_for_condition, data_group_avg.x+error_for_condition, alpha=0.1, color=color)
    axis.title.set_text('bilateral inferior frontal cortex')
axis.set_xlim(0., 0.8)
axis.set_xticks([0., 0.2, 0.4, 0.6, 0.8])
axis.set_xlabel('Time (s)')
axis.set_ylabel('Dipole moment (Am)')
imgA = mpimg.imread('/imaging/hauk/rl05/NocturnalBird/figures/labels/fig_roi_label_inferiorfrontal-lh_low_contrast.png')
imagebox = OffsetImage(imgA, zoom=0.035)  # adjust zoom as needed
ab = AnnotationBbox(
    imagebox,
    xy=(0.15, 1.0),             # upper-right in axis coordinates
    xycoords='axes fraction', # interpret xy as relative to axes
    box_alignment=(1, 1),     # align image top-right
    frameon=False             # no border around image
)
axis.add_artist(ab)
imgA = mpimg.imread('/imaging/hauk/rl05/NocturnalBird/figures/labels/fig_roi_label_inferiorfrontal-rh_low_contrast.png')
imagebox = OffsetImage(imgA, zoom=0.035)  # adjust zoom as needed
ab = AnnotationBbox(
    imagebox,
    xy=(0.3, 1.0),             # upper-right in axis coordinates
    xycoords='axes fraction', # interpret xy as relative to axes
    box_alignment=(1, 1),     # align image top-right
    frameon=False             # no border around image
)
axis.add_artist(ab)

# leg = axis.legend(
#     loc='upper left') 
# leg.get_frame().set_facecolor('none')
# leg.get_frame().set_edgecolor('none')

# add rectangles to demarcate clusters
x_limits = (0.0, 0.8)
span_coords = [(cluster_tstart, cluster_tstop)]
alphas = [0.25]
fh.add_background_spans(ax_dict['A'], span_coords, x_limits, alphas, color=color_cluster)


# 6. PANEL B: Bar plots for the bilateral ATL cluster
axis_bar = ax_dict['B']
x_positions = [1, 1.75, 2.5] # Define positions for the bars

# Loop through each condition to calculate its mean and error for the bar
for j, condition in enumerate(conditions):
    # Select the bilaterally-averaged data for this condition
    cond_data = ds_avg['stcs'][ds_avg['specificity'].isin([condition])]
    
    # Calculate the mean activity within the cluster time window for the bar height
    cond_bar_mean = cond_data.mean(time=(cluster_tstart, cluster_tstop)).x.mean()
    
    # Calculate the average SEM within the cluster time window for the error bar
    # Convert time in seconds to array indices (assuming 250 Hz sampling rate)
    sfreq = 250 
    start_index = int(round(cluster_tstart * sfreq))
    stop_index = int(round(cluster_tstop * sfreq))
    
    # Slice the pre-calculated error array for the current condition and time window
    error_slice = all_errors[j, start_index:stop_index]
    cond_bar_err = error_slice.mean()

    # Plot the bar with its error bar
    axis_bar.bar(x_positions[j], cond_bar_mean, width=0.5, yerr=cond_bar_err, color=colors[j], error_kw={'elinewidth': 1})

# Set the x-axis tick labels to match the conditions
axis_bar.set_xticklabels(conditions)
axis_bar.set_xticks(x_positions)

# Set the title to show the time window in milliseconds
title_tstart_ms = int(round(cluster_tstart * 1000, -1))
title_tstop_ms = int(round(cluster_tstop * 1000, -1))
axis_bar.set_title(f'{title_tstart_ms}-{title_tstop_ms} ms')


fh.label_panels_mosaic(fig, ax_dict, size = 14)


# 7. FINALIZE & SAVE
out_fname = op.join(figures_dir, 'fig4_specificity_biIFG.png')
plt.savefig(out_fname, dpi=300)
plt.close()
print(f"Saved combined figure to {out_fname}.")
