library(tibble)
library(dplyr)
library(readr)

# ---- Configuration ----
data_dir <- '/imaging/hauk/rl05/NocturnalBird/data/logs'
analysis_dir <- '/imaging/hauk/rl05/NocturnalBird/scripts/analysis/behavioural'

# ---- Read and Prepare Data ----
cat("--- Reading and preparing data ---\n")

# Find all subject logfiles, which are CSV files, within the data directory.
all_logfiles <- list.files(path = data_dir, recursive = TRUE, pattern = "\\.csv$", full.names = TRUE)
cat("Excluding sub-16 because they were not included in neural analyses due to bad EEG digitization.\n")
# Exclude a specific subject's data from the list of files.
list_of_logfiles <- all_logfiles[!grepl("sub-16", all_logfiles)]
cat(paste("Found", length(list_of_logfiles), "log files to load.\n"))

# Read all the specified log files into a single data frame.
# The 'id' argument creates a column showing the source file for each row.
group_data <- readr::read_csv(list_of_logfiles, id = "file_name")

# Pre-processing steps to clean and structure the data.
# Select only the columns needed for the analysis.
group_data <- select(group_data, participant, block_nr, trial_nr, item_nr, word1, word2, probe, response, RT, hit, composition, specificity)

# Filter out rows where the 'probe' value is missing, i.e. get trials with a task probe
group_data <- subset(group_data, !is.na(group_data$probe))

# Filter for rows that are specificity trials
group_data <- subset(group_data, !is.na(group_data$specificity))

# Correct the 'hit' values for a specific participant who had inverted button responses.
group_data <- group_data %>%
    mutate(
        hit = if_else(participant == '25', 1 - hit, hit)
    )

# # Exclude participant '33' due to performance at chance level.
# group_data <- group_data %>%
#   filter(participant != '33')

# Convert character columns to factors for statistical modeling.
group_data <- mutate(group_data,
                     participant = factor(participant),
                     item_nr = factor(item_nr),
                     composition = factor(composition),
                     specificity = factor(specificity))

# Convert the data frame to a tibble for better printing and handling.
group_data <- as_tibble(group_data)

cat("Data summary:\n")
print(group_data)


# ---- Write Aggregated Data to a Single CSV ----
cat("\n--- Writing aggregated data to a single CSV file ---\n")
# Define the full path for the output file.
output_file_path <- file.path(analysis_dir, "group_data.csv")
# Save the processed data frame to a CSV file.
# Using readr::write_csv is efficient and doesn't include row numbers by default.
readr::write_csv(group_data, output_file_path)
cat(paste("Aggregated data successfully saved to:", output_file_path, "\n"))
