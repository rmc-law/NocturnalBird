# Fake Diamond: Behavioral Data Analysis (Accuracy & RT)
This repository contains R scripts for analyzing behavioral data from the "fake diamond" experiment. The analysis uses mixed-effects models to investigate the effects of phrase concreteness and denotation on participant response accuracy and reaction times. 

## To run the script
```
# Load the required R module on your server
module load R/4.5.1 

# Execute the script
Rscript 1_group_data.R
Rscript 2_generate_summaries.R
Rscript 3_stats_accuracy.R
Rscript 4_stats_rt.R
```