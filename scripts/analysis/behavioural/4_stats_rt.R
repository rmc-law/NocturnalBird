# ──────────────────────────────────────────────────────────────
# Stats - analyzing RT to comprehension Qs
# in privative and subsective, concrete and abstract phrases
# Author: Ryan Law
# ──────────────────────────────────────────────────────────────

# ---- Setup ----
library(lme4)
library(lmerTest)
library(emmeans)
library(tibble)
library(dplyr)
library(MuMIn)
library(car)

# ---- Configuration ----
data_dir <- '/imaging/hauk/rl05/NocturnalBird/data/logs'
results_dir <- '/imaging/hauk/rl05/NocturnalBird/results/behavioral'
analysis_dir <- '/imaging/hauk/rl05/NocturnalBird/scripts/analysis/behavioural'
group_data_path <- file.path(analysis_dir, "group_data.csv")
models_dir <- file.path(analysis_dir, 'models')
analysis <- 'RT'

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)

cat("--- Reading and preparing data ---\n")
group_data <- readr::read_csv(group_data_path)

# Set factor types
group_data <- group_data %>%
  mutate(
    participant = factor(participant),
    item_nr = factor(item_nr),
    composition = factor(composition),
    specificity = factor(specificity)
  ) %>%
  filter(hit == 1, RT != 0) %>%
  as_tibble()

# filter data
group_data <- group_data %>%
  filter(
    # !participant %in% c('33', '35'),  # remove subjects performing too poorly
    item_nr != 657,                   # remove trial that has low performance across participants
    !is.na(response)                  # remove trials with no responses
  )

group_data <- group_data %>%
  group_by(participant) %>% # Calculate stats for each participant separately
  filter(
    RT < (mean(RT) + 2.5 * sd(RT)), # Remove RTs > 2.5 SDs above the mean
    RT > (mean(RT) - 2.5 * sd(RT))  # Remove RTs < 2.5 SDs below the mean
  ) %>%
  ungroup() # Ungroup for subsequent operations

cat("Data summary:\n")
print(group_data)

# Log-transform RT: better model fit (AIC)
# group_data$log_rt <- log(group_data$RT)

cat("\n--- Analyzing group-level RT ---\n")

# ---- LMER control ----
control_lmer <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 4e5)
)
control_glmer <- glmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 4e5)
)

# ---- Fit LMER for RT ----
model_path_rt <- file.path(models_dir, "final_model_rt.rds")

if (!file.exists(model_path_rt)) {
  cat("\n--- Fitting LMER model for RT ---\n")
  cat("Fitting maximal model: (specificity | participant) + (1 | item_nr)\n")

  m_max_rt <- glmer(RT ~ specificity +
                     (specificity | participant) + (1 | item_nr),
                   data = group_data,
                   family = inverse.gaussian(link = "identity"),
                   control = control_glmer)

  if (isSingular(m_max_rt)) {
    cat("Maximal model is singular. Trying zero-correlation slopes...\n")

    m_zc_rt <- glmer(RT ~ specificity +
                      (specificity || participant) + (1 | item_nr),
                    data = group_data,
                    family = inverse.gaussian(link = "identity"),
                    control = control_lmer)

    if (isSingular(m_zc_rt)) {
      cat("Still singular. Trying only by-participant varying intercepts...\n")

      m1_rt <- glmer(RT ~ specificity +
                      (1 | participant) + (1 | item_nr), 
                    data = group_data,
                    family = inverse.gaussian(link = "identity"),
                    control = control_glmer)

      final_model_rt <- m1_rt

      if (isSingular(final_model_rt)) {
        warning("Even the random-intercepts-only model is singular.")
      }
    } else {
      final_model_rt <- m_zc_rt
    }
  } else {
    final_model_rt <- m_max_rt
  }

  cat("\n--- Final RT Model Chosen ---\n")
  print(formula(final_model_rt))
  saveRDS(final_model_rt, file = model_path_rt)

} else {
  cat("\n--- Loading saved final RT model ---\n")
  final_model_rt <- readRDS(model_path_rt)
  print(formula(final_model_rt))
}

# ---- Summarise final model ----
summary_anova_rt <- Anova(final_model_rt)
r_squared_rt <- r.squaredGLMM(final_model_rt)

cat("\n#### ANOVA Results\n")
print(summary_anova_rt)

cat("\n#### R-squared (Nakagawa & Schielzeth)\n")
print(r_squared_rt)

# ---- Post-hoc Comparisons ----
emm_main_rt <- emmeans(final_model_rt, ~ specificity)

# Back-transform EMMs to original RT scale
emm_main_rt_response <- summary(emm_main_rt, type = "response")  # Gives EMMs in seconds

# Pairwise post-hoc comparisons (on log scale)
pairwise_specificity_rt <- contrast(emm_main_rt, method = "pairwise")

# Back-transformed pairwise comparisons
pairwise_specificity_rt_response <- summary(pairwise_specificity_rt, infer = TRUE, type = "response")


# ---- Print ----
cat("\n#### Estimated Marginal Means (original RT scale, in seconds)\n")
print(emm_main_rt_response)

cat("\n#### Pairwise CONTRASTS for SPECIFICITY (back-transformed to seconds)\n")
print(pairwise_specificity_rt_response)

# ---- Aggregate Results ----
result_text_rt <- c(
  paste("\n\n############################################################"),
  paste("########## RESULTS FOR:", analysis),
  paste("############################################################\n"),
  "\n----- ANOVA Results (Type III with Satterthwaite's method) -----\n",
  capture.output(print(summary_anova_rt)),
  "\n----- R-squared (Nakagawa & Schielzeth) -----\n",
  capture.output(print(r_squared_rt)),
  "\n----- Estimated Marginal Means (original RT scale, in seconds) -----\n",
  capture.output(print(emm_main_rt_response)),
  "\n----- Pairwise CONTRASTS for SPECIFICITY (back-transformed to seconds) -----\n",
  capture.output(print(pairwise_specificity_rt_response))
)

# ---- Write output ----
summary_file_rt <- file.path(results_dir, paste0("stats_", analysis, ".txt"))
writeLines(result_text_rt, con = summary_file_rt)

cat("\n\nAll done. RT summary written to:\n", summary_file_rt, "\n")
