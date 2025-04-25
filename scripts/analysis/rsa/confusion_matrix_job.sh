#!/bin/bash

# Set the partition and other SBATCH specifications for individual subject jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --job-name=confuse_job
#SBATCH --output=/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/rsa/job_log/%j_confuse_output.log
#SBATCH --error=/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/rsa/job_log/%j_confuse_error.log


# Set the subject ID passed as an argument
subject="$subject"
metric="$metric"
#analysis="$analysis" 
#classifier="$classifier"
#data_type="$data_type"


echo "Computing confusion matrix for sub-$subject. Metric: $metric" 

conda activate mne1.9.0

python confusion_matrix_sensors.py -s "$subject" -m "$metric"
