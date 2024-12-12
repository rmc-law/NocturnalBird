#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=decoding_group
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/job_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/job_error.log


# specify decoding analysis and classifier
analysis="$1"
classifier="$2"
data_type="$3"
window="$4"
micro_ave="$5"
generalise="$6"


# read in subjects 
script_dir="/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/"
subjects=($(python -c "import sys; sys.path.append('$script_dir'); from config import subject_ids; print(' '.join(subject_ids))"))


# loop over subjects
for subject in "${subjects[@]}"; do

    if [ "$data_type" = "ROI" ]; then

        rois=("anteriortemporal-lh" "posteriortemporal-lh" "inferiorfrontal-lh" "temporoparietal-lh" "lateraloccipital-lh" )

        for roi in "${rois[@]}"; do

            if [ "$generalise" = "generalise" ]; then
            
                timegen_output_dir="/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output/${analysis}/timegen/${classifier}/${data_type}/${window}/sub-${subject}/${roi}"
                
                if [ ! -e "$timegen_output_dir" ]; then
                    echo "Timegen $data_type $roi output of $analysis does not exist for sub-$subject. Decoding."
                    # sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type",generalise="$generalise",spatial="$spatial" decoding_job.sh
                    sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type",window="$window",micro_ave="$micro_ave",generalise="$generalise",roi="$roi" decoding_job.sh
                else
                    echo "Timegen $data_type output of $analysis exists for sub-$subject. Skipping."
                fi
                
            else
            
                timedecod_output_dir="/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output/${analysis}/diagonal/${classifier}/${data_type}/${window}/sub-${subject}/${roi}"
                
                if [ ! -e "$timedecod_output_dir" ]; then
                    echo "Diagonal $data_type $roi decod output of $analysis does not exist for sub-$subject. Decoding."
                    # sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type",spatial="$spatial" decoding_job.sh
                    sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type",window="$window",micro_ave="$micro_ave",roi="$roi" decoding_job.sh
                else
                    echo "Diagonal $data_type decod output of $analysis exists for sub-$subject. Skipping."
                fi
                
            fi

        done

    else 

        if [ "$generalise" = "generalise" ]; then
        
            timegen_output_dir="/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output/${analysis}/timegen/${classifier}/${data_type}/${window}/sub-${subject}"
            
            if [ ! -e "$timegen_output_dir" ]; then
                echo "Timegen $data_type output of $analysis does not exist for sub-$subject. Decoding."
                # sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type",generalise="$generalise",spatial="$spatial" decoding_job.sh
                sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type",window="$window",micro_ave="$micro_ave",generalise="$generalise" decoding_job.sh
            else
                echo "Timegen $data_type output of $analysis exists for sub-$subject. Skipping."
            fi
            
        else
        
            timedecod_output_dir="/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output/${analysis}/diagonal/${classifier}/${data_type}/${window}/sub-${subject}"
            
            if [ ! -e "$timedecod_output_dir" ]; then
                echo "sub-$subject diagonal decod output-$data_type $analysis $window-does not exist. Decoding."
                # sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type",spatial="$spatial" decoding_job.sh
                sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type",window="$window",micro_ave="$micro_ave" decoding_job.sh
            else
                echo "sub-$subject diagonal decod output-$data_type $analysis $window-exists. Skipping."
                echo "sub-$subject diagonal decod output - $data_type - $analysis - $window - exists. Skipping."
            fi
            
        fi

    fi 

done