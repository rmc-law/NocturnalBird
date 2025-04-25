#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=decoding_group
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/job_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/job_error.log


# specify decoding analysis and classifier
#analysis="$1"
##classifier="$2"
##data_type="$3"
##window="$4"
#micro_ave="$5"
#generalise="$6"


# read in subjects 
script_dir="/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/"
subjects=($(python -c "import sys; sys.path.append('$script_dir'); from config import subject_ids; print(' '.join(subject_ids))"))
models=("specificity" "num_word")

# loop over subjects
for subject in "${subjects[@]}"; do
            
    for model in "${models[@]}"; do
    
        #timegen_output_dir="/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output/${analysis}/timegen/${classifier}/${data_type}/${window}/sub-${subject}/${roi}"
        rsa_output_dir="/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/rsa/output/sub-${subject}/${model}"
        
        if [ ! -e "$rsa_output_dir" ]; then
            echo "RSA output for model $model does not exist for sub-$subject. Performing RSA."
            sbatch --export=subject="$subject",model="$model" rsa_job.sh
        else
            echo "RSA output for model $model exists for sub-$subject. Skipping."
        fi
    
    done

done