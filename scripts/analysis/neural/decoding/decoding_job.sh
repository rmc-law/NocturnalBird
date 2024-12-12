#!/bin/bash

# Set the partition and other SBATCH specifications for individual subject jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=decoding_job
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/%j_decod_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/%j_decod_error.log


# Set the subject ID passed as an argument
subject="$subject"
analysis="$analysis" 
classifier="$classifier"
data_type="$data_type"
window="$window"
micro_ave="$micro_ave"
generalise="$generalise"
roi="$roi"

echo "Decoding $analysis for sub-$subject." 

conda activate mne1.4.2

if [ "$data_type" = "ROI" ]; then

    if [ "$generalise" = "generalise" ]; then

        python decoding.py -s "$subject" -a "$analysis" -clas "$classifier" -data "$data_type" -win "$window" --micro_ave --generalise --roi "$roi"

    else

        python decoding.py -s "$subject" -a "$analysis" -clas "$classifier" -data "$data_type" -win "$window" --micro_ave --roi "$roi"

    fi

else

    if [ "$generalise" = "generalise" ]; then

        python decoding.py -s "$subject" -a "$analysis" -clas "$classifier" -data "$data_type" -win "$window" --micro_ave --generalise

    else

        python decoding.py -s "$subject" -a "$analysis" -clas "$classifier" -data "$data_type" -win "$window" --micro_ave

    fi

fi