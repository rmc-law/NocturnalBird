#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=confusion_group
#SBATCH --output=/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/decoding/job_log/job_output.log
#SBATCH --error=/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/decoding/job_log/job_error.log


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
metrics=("correlation" "euclidean")

# loop over subjects
for subject in "${subjects[@]}"; do
            
    for metric in "${metrics[@]}"; do
    
        #timegen_output_dir="/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output/${analysis}/timegen/${classifier}/${data_type}/${window}/sub-${subject}/${roi}"
        rsa_output_dir="/imaging/hauk/rl05/NocturnalBird/scripts/analysis/neural/rsa/output/sub-${subject}/confusion_matrix/rdm_${metric}.csv"
        
        if [ ! -e "$rsa_output_dir" ]; then
            echo "Confusion matrix for metric $metric does not exist for sub-$subject. Computing."
            sbatch --export=subject="$subject",metric="$metric" confusion_matrix_job.sh
        else
            echo "Confusion matrix for metric $metric exists for sub-$subject. Skipping."
        fi
    
    done

done