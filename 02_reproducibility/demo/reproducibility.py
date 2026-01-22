###############################################################################
# Reproducible Analysis (Python)
# Author: Jared Edgerton
# Date: (fill in)
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
# GitHub workflow (run in Terminal / PowerShell, NOT in Python)
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

# pip install pandas numpy matplotlib statsmodels

import os
import sys
import platform

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels 
import statsmodels.formula.api as smf 

# Python dependency management:
# - Create requirements.txt after everything runs:
#     pip freeze > requirements.txt
# - On another machine:
#     pip install -r requirements.txt

# Reproducible projects should separate:
# - raw data (unchanged inputs)
# - processed data (cleaned outputs)
# - figures and tables (final outputs)

os.makedirs("D:/r_workspace/soda501/soda_501/02_reproducibility/demo/data/raw", exist_ok=True)
os.makedirs("D:/r_workspace/soda501/soda_501/02_reproducibility/demo/data/processed", exist_ok=True)
os.makedirs("D:/r_workspace/soda501/soda_501/02_reproducibility/demo/outputs/figures", exist_ok=True)
os.makedirs("D:/r_workspace/soda501/soda_501/02_reproducibility/demo/outputs/tables", exist_ok=True)

# Logging creates an audit trail:
# - What ran
# - In what order
# - With what parameters
# - Where outputs were written

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[logging.FileHandler("analysis_log.txt", mode="a")]
)

# Pipeline overview:
#   1) Load data
#   2) Save raw data (confirm location)
#   3) Clean data
#   4) Save processed data
#   5) Run three regressions (income as DV)
#   6) Create plot(s)
#   7) Save tables + session info

np.random.seed(123)  # Reproducible randomness for the full pipeline

logging.info("Starting analysis pipeline")

# Expected location for this assignment:
# - data/raw/education_income.csv

logging.info("Loading education/income dataset from data/raw/education_income.csv")

education_income_raw = pd.read_csv("D:/r_workspace/soda501/soda_501/02_reproducibility/demo/data/raw/education_income.csv")

logging.info("Rows loaded: " + str(education_income_raw.shape[0]))
logging.info("Columns loaded: " + str(education_income_raw.shape[1]))

# In many projects, "raw" is treated as read-only and comes from outside.
# Here we re-write it to confirm the exact file used in the run.

logging.info("Saving raw data copy (unchanged)")
# education_income_raw.to_csv("data/raw/education_income.csv", index=False)

# Keep this simple and explicit:
# - Ensure education and income exist
# - Coerce to numeric (if needed)
# - Drop missing
#
# Note: No if/else. If columns are missing, the script will error (which is fine).

logging.info("Cleaning education/income data")

education_income_clean = education_income_raw.copy()
education_income_clean["education"] = pd.to_numeric(education_income_clean["education"])
education_income_clean["income"] = pd.to_numeric(education_income_clean["income"])
education_income_clean = education_income_clean.dropna(subset=["education", "income"])

logging.info("Rows after cleaning: " + str(education_income_clean.shape[0]))

# Create log-income version for Model 3
# If income has zeros or negatives, log(income) is not finite.
education_income_clean["log_income"] = np.log(education_income_clean["income"])

education_income_log = education_income_clean.copy()
education_income_log = education_income_log.replace([np.inf, -np.inf], np.nan)
education_income_log = education_income_log.dropna(subset=["log_income"])

logging.info("Rows with finite log(income): " + str(education_income_log.shape[0]))

logging.info("Saving processed data")
education_income_clean.to_csv("data/processed/cleaned_education_income.csv", index=False)

logging.info("Creating income vs education plot")

plt.figure(figsize=(8, 6))
plt.scatter(education_income_clean["education"], education_income_clean["income"], alpha=0.5)
plt.xlabel("Education (Years)")
plt.ylabel("Income")
plt.title("Income vs Education")
plt.tight_layout()
plt.savefig("outputs/figures/income_education_plot.png")
plt.close()

logging.info("Plot saved to outputs/figures/income_education_plot.png")

logging.info("Fitting Model 1: income ~ education")
model_1 = smf.ols("income ~ education", data=education_income_clean).fit()

logging.info("Fitting Model 2: income ~ education + education^2")
model_2 = smf.ols("income ~ education + I(education**2)", data=education_income_clean).fit()

logging.info("Fitting Model 3: log(income) ~ education (finite log income rows only)")
model_3 = smf.ols("log_income ~ education", data=education_income_log).fit()

# Save model summaries (plain text) for replication checks
logging.info("Saving regression summaries to outputs/tables/")
# TODO: write model summaries to:
#   outputs/tables/model_1_summary.txt
#   outputs/tables/model_2_summary.txt
#   outputs/tables/model_3_summary.txt
with open("outputs/tables/model_1_summary.txt", "w") as f:
    f.write(model_1.summary().as_text())

with open("outputs/tables/model_2_summary.txt", "w") as f:
    f.write(model_2.summary().as_text())

with open("outputs/tables/model_3_summary.txt", "w") as f:
    f.write(model_3.summary().as_text())

# TODO: create and write a regression_coefficients.csv table
coefficients_table = pd.DataFrame({
    "model": ["Model 1: Linear"] * len(model_1.params) +
             ["Model 2: Quadratic"] * len(model_2.params) +
             ["Model 3: Log Income"] * len(model_3.params),
    "term": list(model_1.params.index) + list(model_2.params.index) + list(model_3.params.index),
    "estimate": list(model_1.params) + list(model_2.params) + list(model_3.params),
    "std_error": list(model_1.bse) + list(model_2.bse) + list(model_3.bse),
    "statistic": list(model_1.tvalues) + list(model_2.tvalues) + list(model_3.tvalues),
    "p_value": list(model_1.pvalues) + list(model_2.pvalues) + list(model_3.pvalues)
})

coefficients_table.to_csv("outputs/tables/regression_coefficients.csv", index=False)

# TODO (students):
# - Write session info output to outputs/session_info.txt
with open("outputs/session_info.txt", "w") as f:
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Platform: {platform.platform()}\n")
    f.write(f"NumPy version: {np.__version__}\n")
    f.write(f"Pandas version: {pd.__version__}\n")
    f.write(f"Statsmodels version: {statsmodels.__version__}\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n")

logging.info("Saving session information")
# TODO: write session info to outputs/session_info.txt

# TODO (students):
# - After everything runs, snapshot dependencies.
# - Commit requirements.txt to GitHub.

logging.info("Snapshotting dependencies to requirements.txt")
# TODO: run outside Python:
#   pip freeze > requirements.txt

logging.info("Analysis pipeline completed successfully")
