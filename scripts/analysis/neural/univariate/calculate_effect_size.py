#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:10:41 2024

@author: rl05
"""

import math
import numpy as np
import pandas as pd
import pickle
from eelbrain import Dataset, load, Factor, test, plot
from mne import read_source_estimate, write_labels_to_annot
import config 

subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]
print(f'subjects (n={len(subjects)}): ', subjects)
analysis = input('composition, replicate, denotation: ')


def calculate_cohens_f(condition_means):
    '''
    From this book: https://aaroncaldwell.us/SuperpowerBook/repeated-measures-anova.html
    mu <- c(3.8, 4.2, 4.3)
    sd <- 0.9
    f <- sqrt(sum((mu - mean(mu)) ^ 2) / length(mu)) / sd
    #Cohen, 1988, formula 8.2.1 and 8.2.2
    '''
    std = condition_means.std()
    cohens_f = math.sqrt(np.sum((condition_means - condition_means.mean()) ** 2) / len(condition_means)) / std
    return cohens_f

#%% read in dataset 

if analysis in ['composition','replicate']:
    conditions = ['concrete-baseline','concrete-subsective','abstract-baseline','abstract-subsective']
    region = 'anteriortemporal-lh'
    # colors = plt.cm.Greys([0.4, 0.8])
    # colors_bar = [plt.cm.Blues(0.4),plt.cm.Blues(0.8),plt.cm.Reds(0.4),plt.cm.Reds(0.8)]
    relevant_conditions = ['baseline','subsective']
if analysis == 'denotation': 
    conditions = ['concrete-subsective','concrete-privative','abstract-subsective','abstract-privative']
    region = 'anteriortemporal-lh'
    # colors = plt.cm.YlGn([0.4, 0.8])
    # colors_bar = 
    relevant_conditions = ['subsective','privative']

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
if analysis in ['composition','replicate']:
    composition = [condition.split('-')[1] for condition in conditions_list]
elif analysis == 'denotation': 
    denotation = [condition.split('-')[1] for condition in conditions_list]
ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='oct-6', parc='fake_diamond') 
ds['subject'] = Factor(subjects_list, random=True)
ds['condition'] = Factor(conditions_list)
ds['condition'].sort_cells(conditions)
ds['concreteness'] = Factor(concreteness)
ds['concreteness'].sort_cells(['concrete','abstract'])
if analysis in ['composition','replicate']:
    ds['composition'] = Factor(composition)
    ds['composition'].sort_cells(['baseline','subsective'])
elif analysis == 'denotation': 
    ds['denotation'] = Factor(denotation)
    ds['denotation'].sort_cells(['subsective','privative'])
stc_reset = ds['stcs']


#%% calculate effect size

ds['stcs'] = stc_reset
stc_region = stc_reset.sub(source=region)
ds['stcs'] = stc_region # assign this back to the ds

print ('%s' %region)
print ('Performing a temporal permutation test...')
#use TFCE to control for multiple comparisons
res = testnd.anova(ds['stcs'].mean('source'), 'Association*Structure*Position*Subject', ds=ds, samples=10000, tstart=0, tstop=0.6, pmin=0.05, mintime=0.02, match='Subject') #tfce=0.01,

# subset significant effects
maskSignificant = np.where(res.clusters['p'] <= 0.05, True, False)
effectSignificant = res.clusters[maskSignificant]

if effectSignificant.n_cases != None:
    for i in range(effectSignificant.n_cases):
        cluster = effectSignificant[i]['cluster']
        tstart = (int(effectSignificant[i]['tstart'] * 1000) / 1000)
        tstop = (int(effectSignificant[i]['tstop'] * 1000) / 1000)
        effect = effectSignificant[i]['effect']
        effect = effect.replace(' x ', '%')
        # print (effect)

        # first, calculate subject-specific non-standardised effect sizes
        # subjects = ['A0182']
        raw_differences = []
        word5_means, word6_means, word7_means = [], [], []
        a,b,c,d,e,f = [], [], [], [], [], []

        for s in subjects:

            # get source-averaged timecourse
            timecourse = ds['stcs'].mean('source')

            if effect == 'Structure':
                # get subject-specific, condition-specific, timecourses
                tc_sent = timecourse[ds['Subject'].isin([s]) & ds['Structure'].isin(['sent'])]
                tc_list = timecourse[ds['Subject'].isin([s]) & ds['Structure'].isin(['list'])]

                # get condition specific average
                tc_sent_mean = tc_sent.mean(time=(tstart,tstop)).mean()
                tc_list_mean = tc_list.mean(time=(tstart,tstop)).mean()

                raw_difference = tc_sent_mean - tc_list_mean
                raw_differences.append(raw_difference)

                a.append(tc_sent_mean)
                b.append(tc_list_mean)

                print ('Region %s, %s effect, Subject %s, sent-list raw difference:' %(region, effect, s), raw_difference)

                del tc_sent, tc_list, tc_sent_mean, tc_list_mean, raw_difference

            if effect == 'Association':
                # get subject-specific, condition-specific, timecourses
                tc_high = timecourse[ds['Subject'].isin([s]) & ds['Association'].isin(['high'])]
                tc_low = timecourse[ds['Subject'].isin([s]) & ds['Association'].isin(['low'])]

                # get condition specific average
                tc_high_mean = tc_high.mean(time=(tstart,tstop)).mean()
                tc_low_mean = tc_low.mean(time=(tstart,tstop)).mean()

                raw_difference = tc_high_mean - tc_low_mean
                raw_differences.append(raw_difference)

                a.append(tc_high_mean)
                b.append(tc_low_mean)

                print ('Region %s, %s effect, Subject %s, high-low raw difference:' %(region, effect, s), raw_difference)

                del tc_high, tc_low, tc_high_mean, tc_low_mean, raw_difference

            if effect == 'Position':
                # get subject-specific, condition-specific, timecourses
                tc_word5 = timecourse[ds['Subject'].isin([s]) & ds['Position'].isin(['word5'])]
                tc_word6 = timecourse[ds['Subject'].isin([s]) & ds['Position'].isin(['word6'])]
                tc_word7 = timecourse[ds['Subject'].isin([s]) & ds['Position'].isin(['word7'])]

                # get condition specific average
                tc_word5_mean = tc_word5.mean(time=(tstart,tstop)).mean()
                tc_word6_mean = tc_word6.mean(time=(tstart,tstop)).mean()
                tc_word7_mean = tc_word7.mean(time=(tstart,tstop)).mean()

                word5_means.append(tc_word5_mean)
                word6_means.append(tc_word6_mean)
                word7_means.append(tc_word7_mean)
                # raw_differences.append(raw_difference)

                # print ('Region %s, %s effect, Subject %s, sent-list raw difference:' %(region, effect, s), raw_difference)
                #
                # del tc_sent_word5, tc_sent_word6, tc_sent_word7, tc_list_word5, tc_list_word6, tc_list_word7, tc_sent_word5_mean, tc_sent_word6_mean, tc_sent_word7_mean, tc_list_word5_mean, tc_list_word6_mean, tc_list_word7_mean, raw_difference

            if effect == 'Structure%Position':
                # get subject-specific, condition-specific, timecourses
                tc_sent_word5 = timecourse[ds['Subject'].isin([s]) & ds['Structure'].isin(['sent']) & ds['Position'].isin(['word5'])]
                tc_sent_word6 = timecourse[ds['Subject'].isin([s]) & ds['Structure'].isin(['sent']) & ds['Position'].isin(['word6'])]
                tc_sent_word7 = timecourse[ds['Subject'].isin([s]) & ds['Structure'].isin(['sent']) & ds['Position'].isin(['word7'])]

                tc_list_word5 = timecourse[ds['Subject'].isin([s]) & ds['Structure'].isin(['list']) & ds['Position'].isin(['word5'])]
                tc_list_word6 = timecourse[ds['Subject'].isin([s]) & ds['Structure'].isin(['list']) & ds['Position'].isin(['word6'])]
                tc_list_word7 = timecourse[ds['Subject'].isin([s]) & ds['Structure'].isin(['list']) & ds['Position'].isin(['word7'])]

                # get condition specific average
                tc_sent_word5_mean = tc_sent_word5.mean(time=(tstart,tstop)).mean()
                tc_sent_word6_mean = tc_sent_word6.mean(time=(tstart,tstop)).mean()
                tc_sent_word7_mean = tc_sent_word7.mean(time=(tstart,tstop)).mean()
                tc_list_word5_mean = tc_list_word5.mean(time=(tstart,tstop)).mean()
                tc_list_word6_mean = tc_list_word6.mean(time=(tstart,tstop)).mean()
                tc_list_word7_mean = tc_list_word7.mean(time=(tstart,tstop)).mean()

                a.append(tc_sent_word5_mean)
                b.append(tc_sent_word6_mean)
                c.append(tc_sent_word7_mean)
                d.append(tc_list_word5_mean)
                e.append(tc_list_word6_mean)
                f.append(tc_list_word7_mean)
                # raw_difference = tc_sent_mean - tc_list_mean
                # raw_differences.append(raw_difference)
                #
                # print ('Region %s, %s effect, Subject %s, sent-list raw difference:' %(region, effect, s), raw_difference)
                #
                # del tc_sent_word5, tc_sent_word6, tc_sent_word7, tc_list_word5, tc_list_word6, tc_list_word7, tc_sent_word5_mean, tc_sent_word6_mean, tc_sent_word7_mean, tc_list_word5_mean, tc_list_word6_mean, tc_list_word7_mean, raw_difference

            del timecourse

        if effect == 'Structure' or effect == 'Association':
            # # second, calculate standard deviation
            # raw_differences = np.array(raw_differences)
            # # sigma = raw_differences.std()
            #
            # # finally, calculate cohen's d
            # d = raw_differences.mean() / raw_differences.std()
            # effect_sizes.append(d)
            #
            # print ('\n')
            # print (tstart,'-',tstop,'ms')
            # print ('Cohen\'s d for the %s effect in %s:' %(effect, region), d)

            a = np.array(a)
            b = np.array(b)

            all_means = np.array([a.mean(), b.mean()])
            mu = np.concatenate([a, b])
            f = math.sqrt(np.sum((all_means - all_means.mean()) ** 2) / len(all_means)) / mu.std()

            print ('Cohen\'s f for the %s effect in %s:' %(effect, region), f)


        elif effect == 'Position':
            word5_means = np.array(word5_means)
            word6_means = np.array(word6_means)
            word7_means = np.array(word7_means)
            all_means = np.array([word5_means.mean(), word6_means.mean(), word7_means.mean()])
            mu = np.concatenate([word5_means, word6_means, word7_means])
            f = math.sqrt(np.sum((all_means - all_means.mean()) ** 2) / len(all_means)) / mu.std()

            print ('Cohen\'s f for the %s effect in %s:' %(effect, region), f)

        elif effect == 'Structure%Position':
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            d = np.array(d)
            e = np.array(e)
            f = np.array(f)

            all_means = np.array([a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean()])
            print (all_means)
            mu = np.concatenate([a, b, c, d, e, f])
            f = math.sqrt(np.sum((all_means - all_means.mean()) ** 2) / len(all_means)) / mu.std()

            print ('Cohen\'s f for the %s effect in %s:' %(effect, region), f)

        del a,b,c,d,e,f


