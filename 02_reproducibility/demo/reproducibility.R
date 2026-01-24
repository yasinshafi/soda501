###############################################################################
# Reproducible Analysis (R)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# This script demonstrates:
#   1) Project setup for reproducible research (folders + dependency tracking)
#   2) Logging key steps in an analysis pipeline
#   3) A complete analysis workflow (data -> cleaning -> models -> plots)
#   4) Saving outputs (figures, tables, session info) for replication
#   5) A basic reproducibility check (rerun key steps with same seed)
#   6) (Bonus) Bootstrap simulations with exact replicability
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are explicit so students can follow and modify each piece.
###############################################################################

###############################################################################
# GitHub workflow (run in Terminal / PowerShell, NOT in R)
#
# IMPORTANT:
# - Replace placeholders in <...> with the real URLs / names.
# - You will clone the instructor repo (or workflow repo), then push your work
#   to YOUR OWN GitHub repository for submission.
#
# --- Step A: Clone the instructor repository ---
# 1) Choose (or create) a folder where you want to keep course projects:
#    cd <PATH_TO_YOUR_COURSE_FOLDER>
#
# 2) Clone the instructor repo:
#    git clone <INSTRUCTOR_REPO_URL>
#
# 3) Move into the repo:
#    cd <INSTRUCTOR_REPO_FOLDER_NAME>
#
# 4) (If relevant) move into the reproducibility folder:
#    cd reproducibility
#
# 5) Confirm everything is clean:
#    git status
#
# --- Step B: Create YOUR OWN GitHub repository and push your work ---
# 6) Create a new repo on GitHub (web): e.g., bigdata-ps1-yourname
#
# 7) Check your current branch name:
#    git branch
#
# 8) Add YOUR repo as a remote called "origin" (if origin is not already set):
#    git remote add origin <YOUR_REPO_URL>
#
# 9) Confirm remotes:
#    git remote -v
#
# 10) Stage, commit, and push your changes:
#     git add .
#     git commit -m "PS: reproducible workflow + regressions + plots"
#     git push -u origin main
#
# Notes:
# - If your branch is called "master" instead of "main", use:
#     git push -u origin master
# - If you accidentally cloned with an origin already set to the instructor repo,
#   you can remove and replace it:
#     git remote remove origin
#     git remote add origin <YOUR_REPO_URL>
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.
#
# For teaching: keep installation lines commented out so students can run them
# manually if needed.

# install.packages(c("renv", "logger", "tidyverse", "broom"))

library(renv)       # Dependency management (renv.lock)
library(logger)     # Logging pipeline steps
library(tidyverse)  # Data manipulation + plotting
library(broom)      # Tidy regression outputs (for tables)
library(ggplot2)

# renv::init() creates a project-local library and an renv.lock file.
# Teaching workflow:
# - Run renv::init() ONCE at the start of a project (not every time).
# - After that, use renv::snapshot() to record package versions.
# - On another machine, use renv::restore() to recreate the environment.
#
# NOTE: We leave renv::init() commented out to avoid re-initializing by accident.

# I set a customized path

# file.edit("~/.Renviron") # This opens the file (or creates it if it doesn't exist).
# RENV_PATHS_ROOT=D:/r_workspace/renv # Add this line in the file, save, and restart R

# renv::init()

# Reproducible projects should separate:
# - raw data (unchanged inputs)
# - processed data (cleaned outputs)
# - figures and tables (final outputs)

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)

# Logging creates an audit trail:
# - What ran
# - In what order
# - With what parameters
# - Where outputs were written

logger::log_threshold(DEBUG)
logger::log_appender(appender_file("analysis_log.txt"))

# Pipeline overview:
#   1) Load data
#   2) Save raw data (confirm location)
#   3) Clean data
#   4) Save processed data
#   5) Run three regressions (income as DV)
#   6) Create plot(s)
#   7) Save tables + session info

set.seed(123)  # Reproducible randomness for the full pipeline

log_info("Starting analysis pipeline")

# Expected location for this assignment:
# - data/raw/education_income.csv
log_info("Loading education/income dataset from data/raw/education_income.csv")

education_income_raw <- readr::read_csv("data/raw/education_income.csv") # added this line to load the data

log_info(paste("Rows loaded:", nrow(education_income_raw)))
log_info(paste("Columns loaded:", ncol(education_income_raw)))

# In many projects, "raw" is treated as read-only and comes from outside.
# Here we re-write it to confirm the exact file used in the run.

log_info("Saving raw data copy (unchanged)")
# readr::write_csv(education_income_raw, "data/raw/education_income.csv") # Why do we overwrite?

# Keep this simple and explicit:
# - Ensure education and income exist
# - Coerce to numeric (if needed)
# - Drop missing
#
# Note: No if/else. If columns are missing, the script will error (which is fine).

log_info("Cleaning education/income data")

education_income_clean <- education_income_raw |>
  dplyr::mutate(
    education = as.numeric(education),
    income    = as.numeric(income)
  ) |>
  dplyr::filter(!is.na(education), !is.na(income))

log_info(paste("Rows after cleaning:", nrow(education_income_clean)))

# Create log-income version for Model 3
# If income has zeros or negatives, log(income) is not finite.
education_income_clean <- education_income_clean |>
  dplyr::mutate(log_income = log(income))

education_income_log <- education_income_clean |>
  dplyr::filter(is.finite(log_income))

log_info(paste("Rows with finite log(income):", nrow(education_income_log)))

log_info("Saving processed data")
readr::write_csv(education_income_clean, "data/processed/cleaned_education_income.csv")

log_info("Creating income vs education plot")

ggplot(education_income_clean, aes(x = education, y = income)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(x = "Education (Years)", y = "Income", title = "Income vs Education") +
  theme_minimal()

ggsave("outputs/figures/income_education_plot.png", width = 8, height = 6)

log_info("Plot saved to outputs/figures/income_education_plot.png")


log_info("Fitting Model 1: income ~ education")
model_1 <- lm(income ~ education, data = education_income_clean)

log_info("Fitting Model 2: income ~ education + I(education^2)")
model_2 <- lm(income ~ education + I(education^2), data = education_income_clean)

log_info("Fitting Model 3: log(income) ~ education (finite log income rows only)")
model_3 <- lm(log_income ~ education, data = education_income_log)

# Save model summaries (plain text) for replication checks
log_info("Saving regression summaries to outputs/tables/")
writeLines(capture.output(summary(model_1)), "outputs/tables/model_1_summary.txt")
writeLines(capture.output(summary(model_2)), "outputs/tables/model_2_summary.txt")
writeLines(capture.output(summary(model_3)), "outputs/tables/model_3_summary.txt")
# create and write a regression_coefficients.csv table
coefficients_table <- bind_rows(
  tidy(model_1) |> mutate(model = "Model 1: Linear"),
  tidy(model_2) |> mutate(model = "Model 2: Quadratic"),
  tidy(model_3) |> mutate(model = "Model 3: Log Income")
)
write_csv(coefficients_table, "outputs/tables/regression_coefficients.csv")

# Bootstrap Analysis: Diagnosing the Income-Education Relationship
# Set seed already done at top: set.seed(123)

log_info("Starting bootstrap analysis")

# Number of bootstrap iterations
n_boot <- 1000

# Store bootstrap coefficients
boot_results <- tibble(
  iteration = integer(),
  intercept = numeric(),
  education_coef = numeric()
)

# Bootstrap loop
log_info(paste("Running", n_boot, "bootstrap iterations"))

for (i in 1:n_boot) {
  # Resample rows with replacement
  boot_sample <- education_income_clean |>
    slice_sample(n = nrow(education_income_clean), replace = TRUE)
  
  # Fit model on bootstrap sample
  boot_model <- lm(income ~ education, data = boot_sample)
  
  # Store coefficients
  boot_results <- boot_results |>
    add_row(
      iteration = i,
      intercept = coef(boot_model)[1],
      education_coef = coef(boot_model)[2]
    )
}

log_info("Bootstrap iterations complete")

# Summary statistics
boot_summary <- boot_results |>
  summarise(
    mean_coef = mean(education_coef),
    sd_coef = sd(education_coef),
    ci_lower = quantile(education_coef, 0.025),
    ci_upper = quantile(education_coef, 0.975),
    prop_negative = mean(education_coef < 0)
  )

log_info(paste("Bootstrap 95% CI:", round(boot_summary$ci_lower, 2), "to", round(boot_summary$ci_upper, 2)))
log_info(paste("Proportion of negative coefficients:", round(boot_summary$prop_negative, 3)))

# Save results
write_csv(boot_results, "outputs/tables/bootstrap_coefficients.csv")
write_csv(boot_summary, "outputs/tables/bootstrap_summary.csv")

log_info("Bootstrap results saved to outputs/tables/")

# Plot bootstrap distribution
ggplot(boot_results, aes(x = education_coef)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  geom_vline(xintercept = boot_summary$mean_coef, color = "darkblue") +
  labs(
    x = "Education Coefficient",
    y = "Frequency",
    title = "Bootstrap Distribution of Education Coefficient",
    subtitle = paste("95% CI:", round(boot_summary$ci_lower, 2), "to", round(boot_summary$ci_upper, 2))
  ) +
  theme_minimal()

ggsave("outputs/figures/bootstrap_distribution.png", width = 8, height = 6)

log_info("Bootstrap distribution plot saved")

# TODO (students):
# - Write sessionInfo() output to outputs/session_info.txt


log_info("Saving session information")
writeLines(capture.output(sessionInfo()), "outputs/session_info.txt")


# TODO (students):
# - After everything runs, snapshot dependencies.
# - Commit renv.lock to GitHub.

log_info("Snapshotting dependencies to renv.lock")
renv::snapshot()

log_info("Analysis pipeline completed successfully")
