#!/usr/bin/env python
# coding: utf-8

"""
Sensor-level RSA using mixed sensor types
=========================================

This example demonstrates how to perform representational similarity analysis (RSA) on
MEEG data containing magnetometers, gradiometers and EEG channels. In this scenario
there are important things we need to keep in mind:

1. Different sensor types see the underlying sources from different perspectives, hence
   spatial searchlight patches based on the sensor positions are a bad idea. We will
   perform a searchlight over time only, pooling data from all sensors at all times.
2. The sensors have different units of measurement, hence the numeric data is in
   different orders of magnitude. If we don't compensate for this, only the sensors with
   data in the highest order of magnitude will matter when compuring RDMs. We will
   compute a noise covariance matrix and perform data whitening to achieve this.

The dataset will be the MNE-sample dataset: a collection of 288 epochs in which the
participant was presented with an auditory beep or visual stimulus to either the left or
right ear or visual field.

"""
# sphinx_gallery_thumbnail_number=2

# Import required packages
import operator
import os
import mne
import mne_rsa
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

mne.set_log_level(False)  # Be less verbose


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', type=str, required=True, help='Run subject-specific analysis')
parser.add_argument('-m', '--model', type=str, required=True, help='Specify reference RDM')
# parser.add_argument('-data', '--data_type', type=str, required=True, default='MEEG', help='MEEG or MEG or ROI (source space)')
# parser.add_argument('--roi', type=str, required=False, default=None, help='Perform ROI decoding in this ROI')
args = parser.parse_args()

subject = f'sub-{args.subject}'
print(subject)
model = args.model
print(model)
# data_type = args.data_type
# print('data_type: ', data_type)
print()
    
data_dir = '/imaging/hauk/rl05/fake_diamond/data'
analysis_output_dir = f'/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/rsa/output/{subject}/{model}'
if not os.path.exists(analysis_output_dir):
    os.makedirs(analysis_output_dir, exist_ok=True)
preprocessed_data_path = os.path.join(data_dir, 'preprocessed')

start_time = time.time()


if model == 'specificity':
    conditions = ['low','mid','high']
elif model == 'num_word':
    conditions = ['word','phrase']

    
print('Reading in epochs.')
epochs = mne.read_epochs(os.path.join(data_dir,'preprocessed',subject,'epoch',f'{subject}_epo.fif'))

if model == 'num_word':
    epochs = mne.epochs.combine_event_ids(epochs, ['low/word1', 'high/word1'], {'word': 1})
    epochs = mne.epochs.combine_event_ids(epochs, ['mid/word1'], {'phrase': 2})

epochs = epochs[conditions]

print('Saving evoked plots.')
for condition in conditions: # plot condition specific evokeds
    fig_evoked = epochs[condition].average().plot()
    fig_evoked_fname = os.path.join(analysis_output_dir, f'fig_evoked_{condition}.png')
    fig_evoked.savefig(fig_evoked_fname)
    plt.close(fig_evoked)
fig_evokeds = epochs.average().plot() # plot grand average
fig_evokeds_fname = os.path.join(analysis_output_dir, 'fig_evoked_all.png')
fig_evokeds.savefig(fig_evokeds_fname)
plt.close(fig_evokeds)


# need to normalize signal amplitude across channel types (we have eeg+meg)
print('Computing noise covariance.')
noise_cov = mne.compute_covariance(
    epochs, tmin=-0.5, tmax=-0.3, method='auto', rank='info'
)
# noise_cov.plot(epochs.info)

#%%#######################################################################################
# compute reference RDM

if model == 'specificity':    
    epochs = mne.concatenate_epochs([epochs["low"], epochs["mid"], epochs["high"]])
elif model == 'num_word':    
    epochs = mne.concatenate_epochs([epochs['word'], epochs['phrase']])


print('Computing model RDM.')
model_rdm = mne_rsa.compute_rdm(epochs.events[:, 2], metric=operator.ne)
mne_rsa.plot_rdms(model_rdm)

#%% Perform RSA across time

print(f'Starting RSA. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')

rsa_scores = mne_rsa.rsa_epochs(
    epochs,
    model_rdm,
    noise_cov=noise_cov,
    temporal_radius=0.02,
    y=np.arange(len(epochs)),
)


# save rsa scores
rsa_scores_fname = os.path.join(analysis_output_dir, f'rsa_scores.npy')
np.save(rsa_scores_fname, rsa_scores.data)

# save rsa fig
fig_rsa = rsa_scores.plot(units=dict(misc="Spearman correlation"))
fig_output_fname = os.path.join(analysis_output_dir, f'fig_rsa_corr.png')
fig_rsa.savefig(fig_output_fname)

print(f'Finished RSA. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')
