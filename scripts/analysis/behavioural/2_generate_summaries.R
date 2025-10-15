library(dplyr)
library(ggplot2)
library(readr)

# --- Configuration ---
summaries_dir <- '/imaging/hauk/rl05/NocturnalBird/results/behavioral'
aggregated_data_path <- '/imaging/hauk/rl05/NocturnalBird/scripts/analysis/behavioural/group_data.csv'

# Create a subdirectory for plots to keep things organized.
summaries_dir <- file.path(summaries_dir, "summaries")
if (!dir.exists(summaries_dir)) {
  dir.create(summaries_dir, recursive = TRUE)
}

# --- Load Data ---
cat("--- Loading aggregated data ---\n")
# Check if the aggregated data file exists before trying to load it.
if (!file.exists(aggregated_data_path)) {
  stop("Aggregated data file not found at: ", aggregated_data_path,
       "\nPlease run the previous script to generate it.")
}
group_data <- readr::read_csv(aggregated_data_path)
cat("Data loaded successfully.\n")
print(head(group_data))


# --- Calculate Descriptive Statistics ---
cat("\n--- Calculating and saving descriptive statistics ---\n")

# Summarize accuracy (hit rate) and reaction time (RT) by each condition.
# We will calculate the mean, standard deviation (sd), and count (n).
# We also calculate the standard error of the mean (sem).
# Note: For accuracy summary, we should use the full dataset, not just hit == 1
summary_stats <- group_data %>%
  group_by(composition, specificity) %>%
  summarise(
    mean_accuracy = mean(hit),
    sd_accuracy = sd(hit),
    n_total = n(),
    .groups = 'drop'
  )

rt_stats <- group_data %>%
  # Only include correct trials for RT analysis
  filter(hit == 1) %>% 
  group_by(composition, specificity) %>%
  summarise(
    mean_rt = mean(RT, na.rm = TRUE),
    sd_rt = sd(RT, na.rm = TRUE),
    n_correct = n(),
    sem_rt = sd_rt / sqrt(n_correct),
    .groups = 'drop' # Drop grouping structure after summarising
  )

# Join the accuracy and RT stats together
descriptive_stats <- full_join(summary_stats, rt_stats)

# Display the summary table in the console.
cat("Descriptive Statistics:\n")
print(descriptive_stats)

# Save the summary statistics to a CSV file.
summary_output_path <- file.path(summaries_dir, "descriptive_statistics.csv")
readr::write_csv(descriptive_stats, summary_output_path)
cat(paste("Descriptive statistics saved to:", summary_output_path, "\n"))




full_summary_by_condition <- group_data %>%
  group_by(specificity) %>%
  summarise(
    # -- Accuracy Statistics (calculated on ALL trials for this group) --
    n_total = n(),
    accuracy = mean(hit),
    
    # -- Reaction Time (RT) Statistics (calculated ONLY on correct trials) --
    # The trick is to filter the data *inside* the summary function.
    n_correct = sum(hit),
    mean_rt = mean(RT[hit == 1], na.rm = TRUE),
    sd_rt = sd(RT[hit == 1], na.rm = TRUE),
    median_rt = median(RT[hit == 1], na.rm = TRUE),
    
    # .groups = 'drop' is good practice to prevent the output from remaining grouped.
    .groups = 'drop' 
  )

# Print the comprehensive summary to the console
print(full_summary_by_condition)

# Save this single, powerful summary to a CSV file
write.csv(full_summary_by_condition, file = file.path(summaries_dir, "full_summary_by_condition.csv"), row.names = FALSE)


# --- 2. Summarizing by Participant ---
# This is a great way to check for outliers or problematic participants.

cat("\n--- Generating summary by participant ---\n")

summary_by_participant <- group_data %>%
  group_by(participant) %>%
  summarise(
    n_trials = n(),
    accuracy = mean(hit),
    mean_rt_correct = mean(RT[hit == 1], na.rm = TRUE),
    .groups = 'drop'
  )

print(summary_by_participant)
write.csv(summary_by_participant, file = file.path(summaries_dir, "summary_by_participant.csv"), row.names = FALSE)


# --- 3. Summarizing by Item ---
# This is crucial for checking if any specific items are behaving strangely (e.g., are too hard or too easy).
# It's often useful to group by both item and condition.

cat("\n--- Generating summary by item ---\n")

summary_by_item <- group_data %>%
  group_by(item_nr, specificity) %>%
  summarise(
    n_responses = n(),
    accuracy = mean(hit),
    mean_rt_correct = mean(RT[hit == 1], na.rm = TRUE),
    .groups = 'drop'
  )

print(summary_by_item)
write.csv(summary_by_item, file = file.path(summaries_dir, "summary_by_item.csv"), row.names = FALSE)



# # --- Generate and Save Plots ---
# cat("\n--- Generating and saving plots ---\n")

# # 1. Bar plot for Mean Accuracy by Condition
# accuracy_plot <- ggplot(descriptive_stats, aes(x = condition, y = mean_accuracy, fill = condition)) +
#   geom_bar(stat = "identity", position = "dodge", color = "black") +
#   geom_text(aes(label = round(mean_accuracy, 2)), vjust = -0.5, position = position_dodge(0.9), size = 3.5) +
#   scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
#   facet_wrap(~composition) + # Facet by composition to see interactions
#   labs(
#     title = "Mean Accuracy by Condition",
#     subtitle = "Faceted by Composition Type",
#     x = "Condition",
#     y = "Mean Accuracy (Hit Rate)"
#   ) +
#   theme_minimal(base_size = 14) +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

# # Save the accuracy plot using a non-interactive method to avoid X11 errors
# accuracy_plot_path <- file.path(summaries_dir, "accuracy_by_condition.jpg")
# CairoJPEG(filename = accuracy_plot_path, width = 10, height = 8, units = "in", res = 300)
# print(accuracy_plot) # Explicitly print the plot to the device
# dev.off() # Close the graphics device
# cat(paste("Accuracy plot saved to:", accuracy_plot_path, "\n"))


# # 2. Boxplot for Reaction Times (for correct trials) by Condition
# rt_plot <- ggplot(filter(group_data, hit == 1), aes(x = condition, y = RT, fill = condition)) +
#   geom_boxplot() +
#   facet_wrap(~composition) + # Facet by composition
#   labs(
#     title = "Distribution of Reaction Times for Correct Trials",
#     subtitle = "Faceted by Composition Type",
#     x = "Condition",
#     y = "Reaction Time (RT)"
#   ) +
#   theme_minimal(base_size = 14) +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

# # Save the RT plot using a non-interactive method
# rt_plot_path <- file.path(summaries_dir, "rt_by_condition_boxplot.png")
# png(rt_plot_path, width = 10, height = 8, units = "in", res = 300)
# print(rt_plot) # Explicitly print the plot to the device
# dev.off() # Close the graphics device
# cat(paste("Reaction time plot saved to:", rt_plot_path, "\n"))
