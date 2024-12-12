#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:16:38 2024

@author: rl05

Perform cluster-based permutation tests on ROI data
"""


import os
import os.path as op
import sys
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mne import read_source_estimate
from eelbrain import Dataset, load, Factor, plot, testnd, test, concatenate
from eelbrain._stats.stats import variability

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config

subjects = [subject for subject in config.subject_ids if subject not in ['16']]
print(f'subjects (n={len(subjects)}): ', subjects)

data_dir = op.join(config.project_repo, 'data')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

ch_type = 'MEEG'
analysis = 'specificity'
parc = 'semantics'
test = 't' # or st for spatiotemporal test
source_space = 'oct-6' # ico-4 or oct-6 (ico4 for spatiotemporal test)

def calculate_cohens_f(observations_per_condition):
    '''
    From this book: https://aaroncaldwell.us/SuperpowerBook/repeated-measures-anova.html
    mu <- c(3.8, 4.2, 4.3)
    sd <- 0.9
    f <- sqrt(sum((mu - mean(mu)) ^ 2) / length(mu)) / sd
    #Cohen, 1988, formula 8.2.1 and 8.2.2
    '''
    condition_means = np.array([condition.mean() for condition in observations_per_condition])
    std = np.concatenate(observations_per_condition).std()
    cohens_f = math.sqrt(np.sum((condition_means - condition_means.mean()) ** 2) / len(condition_means)) / std
    return cohens_f


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

#%% make eelbrain dataset

stcs = []
subject_list = []
specificity_list = []
region_list = []
hemi_list = []
ds = Dataset()

# naming = [('anteriortemporal','bilateralATL'),('inferiorfrontal','bilateralIFG')]
if analysis == 'specificity':
    levels_specificity = ['low','mid','high']
    levels_hemisphere = ['lh','rh']
    levels_region = ['anteriortemporal']#, 'posteriortemporal','inferiorfrontal', 'temporoparietal']

for subject in subjects:
    subject = f'sub-{subject}'
    print(f'Reading in source estimates from {subject}.')
    stc_path = op.join(data_dir, 'stcs', subject)
    for hemi in levels_hemisphere:
        for region in levels_region:
            for spec in levels_specificity:
                if source_space == 'ico-4':
                    stc = read_source_estimate(op.join(stc_path, f'{subject}_{spec}_{ch_type}_ico4-lh.stc'), subject='fsaverage_src')
                elif source_space == 'oct-6':
                    stc = read_source_estimate(op.join(stc_path, f'{subject}_{spec}_{ch_type}-lh.stc'), subject='fsaverage_src')
                # convert mne stc into eelbrain ndvar
                stc = load.mne.stc_ndvar(stc, subject='fsaverage_src', src=source_space, parc=parc)
                # subset to region and get time course
                roi_name = region + '-' + hemi
                stc = stc.sub(source=roi_name).mean('source')
                stcs.append(stc)
                subject_list.append(subject)
                hemi_list.append(hemi)
                region_list.append(region)
                specificity_list.append(spec)

if analysis == 'specificity':
    ds['subject'] = Factor(subject_list,random=True)
    ds['stcs'] = concatenate(stcs, dim='case')
    ds['hemisphere'] = Factor(hemi_list)
    ds['region'] = Factor(region_list)
    ds['specificity'] = Factor(specificity_list)
    ds['specificity'].sort_cells(['low','mid','high'])
# elif analysis == 'specificity_word':
#     ds['specificity'] = Factor(condition_list)
#     ds['specificity'].sort_cells(['low','high'])
stc_reset = ds['stcs']
print(f'Read in datasets from n={len(subjects)} subjects.')

#%% run roi test

# rois = ['anteriortemporal-lh', 'posteriortemporal-lh','inferiorfrontal-lh', 'temporoparietal-lh',
#         'anteriortemporal-rh', 'posteriortemporal-rh','inferiorfrontal-rh', 'temporoparietal-rh']

if analysis.startswith('specificity'):
    x = 'specificity*hemisphere*subject'
if test == 'st':
    test_suffix = '_st'
else:
    test_suffix = ''   

if test == 't':
    results_dir = op.join(config.project_repo, 'results/neural/roi/anova/bilateralATL')            

    # hemi_data = []
    # for hemi in ['lh','rh']:
    #     ds_stcs_hemi = stc_reset
    #     roi_name = roi + '-' + hemi
    #     stc_region_hemi = ds_stcs_hemi.sub(source=roi_name) # get hemi-specific region data 
    #     print(stc_region_hemi)
    #     ds_stcs_hemi = stc_region_hemi.mean('source')
    #     hemi_data.append(ds_stcs_hemi)
    # test = concatenate(hemi_data, dim='case')
        
    # ds['stcs'] = stc_reset
    # stc_region = stc_reset.sub(source=roi) # subset language network region data
    # ds['stcs'] = stc_region # assign this back to the ds
    
    # perform temporal permutation test in a particular region        
    res = testnd.ANOVA(y=ds['stcs'], 
                        x=x, 
                        ds=ds, 
                        samples=5000, 
                        pmin=0.05,
                        tstart=0.6,
                        tstop=1.4,
                        match='subject')
    pickle.dump(res, open(op.join(results_dir, 'specificity*hemisphere.pickle'), 'wb'))

    f = open(op.join(results_dir, f'{analysis}_{ch_type}_results_table.txt'), 'w')
    f.write('Model: %s, N=%s\n' %(res.x, len(subjects)))
    f.write('tstart=%s, tstop=%s, samples=%s, pmin=%s, mintime=??\n\n' %(res.tstart, res.tstop, res.samples, res.pmin))
    f.write(str(res.clusters))
    f.close()
    
    pmin = 0.05
    mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
    sign_clusters = res.clusters[mask_sign_clusters]

    if sign_clusters.n_cases != None: #check for significant clusters
        for i in range(sign_clusters.n_cases):
            cluster_nb = i+1
            cluster = sign_clusters[i]['cluster']
            tstart = sign_clusters[i]['tstart']
            tstop = sign_clusters[i]['tstop']
            effect = sign_clusters[i]['effect']
            pval = sign_clusters[i]['p']
            effect = effect.replace(' x ', '%')

            print('Plotting time series.')
            timecourse = ds['stcs']
            # p = plot.Style(linewidth=4)
            # colors = plot.colors_for_oneway(['low','mid','high'])
            colors = plt.cm.Oranges([0.45, 0.65, 0.8]).tolist()
            activation = plot.UTSStat(timecourse, 
                                      effect, 
                                      ds=ds, 
                                      error='sem', 
                                      match='subject', 
                                      # legend='upper right', 
                                      legend=False,
                                      xlabel='Time (s)', 
                                      ylabel='Current (Am)', 
                                      xlim=(0.6,1.4), 
                                      bottom=1.3e-11,
                                      # colors=plt.cm.Oranges([0.4, 0.6, 0.8]).tolist(),
                                      colors=colors,
                                      error_alpha=0.15,
                                      h=3,
                                      w=5,
                                      tight=True,
                                      clip=True)
                                      # title=f'Cluster {i+1}: Effect of {effect} at ATL, pval={pval}',
                                      # clusters=res.clusters)
            # legend = plot.ColorList(colors)
            # legend.save(op.join(results_dir, f'fig_cluster{i+1}_timecourse{test_suffix}_colorlist.png'))
            activation.add_vspan(xmin=tstart, xmax=tstop, color='yellow', zorder=-50, alpha=0.2)
            # activation.add_hline(y=1.4e-11, xmin=tstart*1000, xmax=tstop*1000, color='purple', alpha=0.8)
            activation.save(op.join(results_dir, f'fig_cluster{i+1}_timecourse{test_suffix}.png'))
            activation.close()

            ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop)) #!!! should be restrained to my sign timewindow
            bar = plot.Barplot(ds['average_source_activation'], 
                               effect, 
                               ds=ds, 
                               colors=colors,
                               # title=f'Cluster {i+1}: {round(tstart,3)}-{round(tstop,3)}, ATLs, pval={pval}', 
                               match='subject', 
                               # frame=None,
                               h=3,
                               w=2.5,
                               tight=True,
                               ylabel='Current (Am)')
            bar.save(op.join(results_dir, f'fig_cluster{i+1}_bar_effect-{effect}{test_suffix}.png'))
            bar.close()
            
            # split conditions
            if not analysis.startswith('specificity'):
                if analysis == 'composition':
                    effect_split = 'concreteness%composition'
                elif analysis == 'denotation':
                    effect_split = 'concreteness%denotation'
                elif analysis == 'specificity':
                    effect_split = 'specificity'
                bar = plot.Barplot(ds['average_source_activation'], 
                                effect_split, 
                                ds=ds, 
                                title=f'Cluster {i+1}: {round(tstart,3)}-{round(tstop,3)}, ATLs, pval={pval}', 
                                match='subject', 
                                ylabel='Activity (MNE)')
                bar.save(op.join(results_dir, f'fig_cluster{i+1}_bar_effect-split-{effect_split}{test_suffix}.png'))
                bar.close()
                
            # # calculate average F-values
            # if effect == 'specificity':
            #     cluster_f = res.f[0]
            # elif effect == 'hemi':
            #     cluster_f = res.f[1]
            # elif effect == 'specificity%hemi':
            #     cluster_f = res.f[2]
            # cluster_mean_f = cluster_f.mean(time=(tstart,tstop))

            # calculate effect size
            roi_activity = timecourse.mean(time=(tstart,tstop))
            if effect == 'specificity':
                specificity_levels = ['low','mid','high']
                hemi_levels = ['lh','rh']
            # elif effect == 'specificity_word':
                # effect_conditions = ('low','high')
            if effect == 'specificity%hemi':
                
            roi_activity_per_condition = [roi_activity[ds[effect].isin([effect_conditions[0]])],
                                          roi_activity[ds[effect].isin([effect_conditions[1]])]]
            cohens_f = calculate_cohens_f(roi_activity_per_condition)
            print(f'Cohen\'s f for {analysis} in {roi} cluster {i+1}: {cohens_f}', file=open(op.join(output_dir, f'effect_size_cluster{i+1}.txt'), 'w'))


#%% plot figure - bilateral ATL effect
    
# figure parameters
fig_dir = op.join(config.project_repo, 'figures', 'univariate')
n_subplots = 1
fig_name = op.join(fig_dir, f'{analysis}_timecourse_bilateralATL.png')
fig_height = 2.5
fig_width = 5 # 10 for two subplots sidebyside
fig, axes = plt.subplots(1, n_subplots, sharex=True, sharey=True, dpi=300)
fig.set_size_inches(fig_width, fig_height)
axis = axes

# output_dir = op.join(results_dir, , analysis)

relevant_conditions = levels_specificity
colors = plt.cm.Oranges([0.4, 0.6, 0.8]).tolist()
ds['stcs']= stc_reset
times = ds['stcs'].time
# plot condition time courses with 1 within-subjects sem
for relevant_condition, color in zip(relevant_conditions,colors):
    data = ds['stcs'][ds[analysis].isin([relevant_condition])]
    data_group_avg = data.mean('case') # average over subjects
    error = variability(y=data.x[:,:], x=ds['specificity'], match=ds['subject'], pool=True, spec='sem')
    axis.plot(times, data_group_avg.x, color=color, lw=3)
    axis.fill_between(times, data_group_avg.x-error, data_group_avg.x+error, alpha=0.2, color=color)
# axis.set_xlim(0., 0.8)

# more figure parameters
xticks = [0., 0.2, 0.4, 0.6, 0.8]
plt.xticks(xticks)
axis.set_xlabel('Time (s)')
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
axis.spines['left'].set_visible(False)
axis.spines['bottom'].set_visible(True)
axis.set_ylabel('Current (Am)')
    
# add custom legend
legend_elements = [Line2D([0], [0], color=color, lw=3) for color in colors]
# if analysis == 'specificity_word': 
#     fig.get_axes()[0].legend(legend_elements, ['low', 'high'], loc='upper right')

# read in permutation test pickle file
print('Unloading pickle for bilateral ATLs.')
pickle_fname = op.join(results_dir, 'specificity*hemisphere.pickle')
with open(pickle_fname, 'rb') as f:
    res = pickle.load(f)
pmin = 0.05
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

        # output_dir = op.join(results_dir, roi, 'replicateB&P')
        # if not op.exists(output_dir):
            # os.makedirs(output_dir, exist_ok=True)
        roi_activity = ds['stcs'].mean(time=(cluster_tstart,cluster_tstop))
        
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
        if analysis == 'specificity':
            x_positions = [1,1.5,2,2.5]
            for d_label, d in zip(['low','mid', 'high'],['low','mid','high']):
                cond = ds['stcs'][ds['specificity'].isin([d])]
                error = variability(y=cond.x[:,:], x=ds['specificity'], match=ds['subject'], pool=True, spec='sem')
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
        fig_bar.savefig(op.join(fig_dir, f'{cluster_effect}_bilateralATL_c{i+1}_split-bars.png'))
        plt.close(fig_bar)

plt.tight_layout()
fig.savefig(fig_name)
plt.close()

