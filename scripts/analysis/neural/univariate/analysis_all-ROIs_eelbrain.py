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

from mne import read_source_estimate
from eelbrain import Dataset, load, Factor, plot, testnd, test

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config

subjects = [subject for subject in config.subject_ids if subject not in ['16']]
print(f'subjects (n={len(subjects)}): ', subjects)

data_dir = op.join(config.project_repo, 'data')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir
os.environ['DISPLAY'] = 'localhost:11.0'

ch_type = 'MEEG'
analysis = 'specificity'
parc = 'semantics' # semantics
test = 't' # t or st
source_space = 'oct6' #ico4 or oct6 (ico4 for spatiotmeporal test)

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
    'ytick.major.width': LINEWIDTH
})

#%% make eelbrain dataset

stcs = []
subject_list = []
condition_list = []
ds = Dataset()

if analysis == 'specificity':
    conditions = ['low','mid','high']
elif analysis == 'specificity_word':
    conditions = ['low','high']

for subject in subjects:
    subject = f'sub-{subject}'
    print(f'Reading in source estimates from {subject}.')
    stc_path = op.join(data_dir, 'stcs', subject)
    for condition in conditions:
        condition = condition.replace('/','-')
        if source_space == 'ico4':
            stc = read_source_estimate(op.join(stc_path, f'{subject}_{condition}_{ch_type}_ico4-lh.stc'), subject='fsaverage_src')
        elif source_space == 'oct6':
            stc = read_source_estimate(op.join(stc_path, f'{subject}_{condition}_{ch_type}-lh.stc'), subject='fsaverage_src')
        stcs.append(stc)
        subject_list.append(subject)
        condition_list.append(condition)

if analysis.startswith('specificity'):
    specificity = [condition.split('-')[0] for condition in condition_list if condition.startswith(('low','mid','high'))]

if test == 't':
    ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='oct-6', parc=parc) 
if test == 'st':
    ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='ico-4', parc=parc) 
ds['subject'] = Factor(subject_list,random=True)

if analysis == 'specificity':
    ds['specificity'] = Factor(specificity)
    ds['specificity'].sort_cells(['low','mid','high'])
elif analysis == 'specificity_word':
    ds['specificity'] = Factor(specificity)
    ds['specificity'].sort_cells(['low','high'])
stc_reset = ds['stcs']
print(f'Read in datasets from n={len(subjects)} subjects.')

#%% run roi test

rois = ['anteriortemporal-lh', 'posteriortemporal-lh','inferiorfrontal-lh', 'temporoparietal-lh',
        'anteriortemporal-rh', 'posteriortemporal-rh','inferiorfrontal-rh', 'temporoparietal-rh']

x = 'specificity*subject'

results_dir = '/imaging/hauk/rl05/NocturnalBird/results/neural/roi/anova'       
for roi in rois:
    print(roi)
    output_dir = op.join(results_dir, roi, analysis)
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    ds['stcs'] = stc_reset
    stc_region = stc_reset.sub(source=roi) # subset language network region data
    ds['stcs'] = stc_region # assign this back to the ds
    
    # perform temporal permutation test in a particular region        
    res = testnd.ANOVA(y=ds['stcs'].mean('source'), 
                        x=x, 
                        ds=ds, 
                        samples=5000, 
                        pmin=0.05,
                        tstart=0.6,
                        tstop=1.4,
                        match='subject')
    pickle.dump(res, open(op.join(output_dir, f'{roi}.pickle'), 'wb'))

    f = open(op.join(output_dir, f'{analysis}_{ch_type}_{roi}_results_table.txt'), 'w')
    f.write('Model: %s, N=%s\n' %(res.x, len(subjects)))
    f.write('tstart=%s, tstop=%s, samples=%s, pmin=%s, mintime=??\n\n' %(res.tstart, res.tstop, res.samples, res.pmin))
    f.write(str(res.clusters))
    f.close()
    
    pmin = 0.1
    mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
    sign_clusters = res.clusters[mask_sign_clusters]

    if test == 'st':
        test_suffix = '_st'
    else:
        test_suffic = ''
    if sign_clusters.n_cases != None: #check for significant clusters
        for i in range(sign_clusters.n_cases):
            cluster_nb = i+1
            cluster = sign_clusters[i]['cluster']
            tstart = sign_clusters[i]['tstart']
            tstop = sign_clusters[i]['tstop']
            effect = sign_clusters[i]['effect']
            pval = sign_clusters[i]['p']
            effect = effect.replace(' x ', '%')

            print('Plotting time series for %s' %roi)
            timecourse = stc_region.mean('source')
            activation = plot.UTSStat(timecourse, 
                                      effect, 
                                      ds=ds, 
                                      error='sem', 
                                      match='subject', 
                                      legend='lower left', 
                                      xlabel='Time (ms)', 
                                      ylabel='Activity (MNE)', 
                                      xlim=(0.6,1.4), 
                                      title=f'Cluster {i+1}: Effect of {effect} at {roi}, pval={pval}')
            activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50, alpha=0.4)
            activation.save(op.join(output_dir, f'fig_cluster{i+1}_timecourse.png'))
            activation.close()

            ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop)) #!!! should be restrained to my sign timewindow
            bar = plot.Barplot(ds['average_source_activation'], 
                               effect, 
                               ds=ds, 
                               title=f'Cluster {i+1}: {round(tstart,3)}-{round(tstop,3)}, {roi}, pval={pval}', 
                               match='subject', 
                               ylabel='Activity (MNE)')
            bar.save(op.join(output_dir, f'fig_cluster{i+1}_bar_effect-{effect}.png'))
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
                                title=f'Cluster {i+1}: {round(tstart,3)}-{round(tstop,3)}, {roi}, pval={pval}', 
                                match='subject', 
                                ylabel='Activity (MNE)')
                bar.save(op.join(output_dir, f'fig_cluster{i+1}_bar_effect-split-{effect_split}.png'))
                bar.close()
                

            # calculate effect size & mean F values in clusters
            roi_activity = timecourse.mean(time=(tstart,tstop))
            if effect == 'specificity':
                
                cluster_mean_f = res.f[0].mean(time=(tstart,tstop))
                print(f'Mean F-values\'s f for {analysis} in {roi} cluster {i+1}: {cluster_mean_f}', file=open(op.join(output_dir, f'cluster_mean_F_cluster{i+1}.txt'), 'w'))

                effect_conditions = ('low','mid','high')
                roi_activity_per_condition = [roi_activity[ds[effect].isin([effect_conditions[0]])],
                                              roi_activity[ds[effect].isin([effect_conditions[1]])],
                                              roi_activity[ds[effect].isin([effect_conditions[2]])]]
                cohens_f = calculate_cohens_f(roi_activity_per_condition)
                print(f'Cohen\'s f for {analysis} in {roi} cluster {i+1}: {cohens_f}', file=open(op.join(output_dir, f'effect_size_cluster{i+1}.txt'), 'w'))
