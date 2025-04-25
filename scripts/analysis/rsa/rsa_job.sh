#!/bin/bash

# Set the partition and other SBATCH specifications for individual subject jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=rsa_job
#SBATCH --output=/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/rsa/job_log/%j_rsa_output.log
#SBATCH --error=/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/rsa/job_log/%j_rsa_error.log


# Set the subject ID passed as an argument
subject="$subject"
model="$model"
#analysis="$analysis" 
#classifier="$classifier"
#data_type="$data_type"


echo "Running RSA for sub-$subject. Model: $model" 

conda activate mne1.9.0

python rsa.py -s "$subject" -m "$model"
