import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from mne import read_epochs

import config 

subjects = config.subject_ids
data_dir = op.join(config.project_repo, 'data')

preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]
print(f'subjects (n={len(subjects)}): ', subjects)


list_percent_dropped = []

print('epochs dropped:')
for subject in subjects:
    epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
    epoch_fname = op.join(epoch_path, f'{subject}_epo.fif')
    epochs = read_epochs(epoch_fname, preload=False, verbose=False)
    percent_dropped = round(100-(len(epochs)/900*100),2)
    list_percent_dropped.append(percent_dropped)
    print(f'{subject}: {len(epochs)} / 900 ({percent_dropped}%)')

mean_num_dropped = np.array(list_percent_dropped).mean()
std_num_dropped = np.array(list_percent_dropped).std()
print('Average percent of epochs dropped: ', round(mean_num_dropped,2))
print('Standard deviation percent of epochs dropped: ', round(std_num_dropped,2))