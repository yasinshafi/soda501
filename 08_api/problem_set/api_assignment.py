###############################################################################
# SoDA 501 – Week 8: APIs and Election Forecasting
# Author: Yasin Shafi
# Date: 2026-03-05
#
# Tasks:
#   1) Connect to FRED API via environment variable
#   2) Replicate the baseline forecasting pipeline
#   3) Build an improved out-of-sample forecaster (hold out 2020)
#   4) Compare baseline vs improved model with a table and figure
# Your working directory needs to be soda_501/08_api
###############################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fredapi import Fred

# --------------------------------------------------------------------------------
# Task 1: Load FRED API key from environment variable and Confirm API call works
# --------------------------------------------------------------------------------
# Set the environment variable before running:
#   export FRED_API_KEY="your_key_here"   (Linux/Mac)
#   set FRED_API_KEY=your_key_here        (Windows CMD)

fred_api_key = os.environ.get("FRED_API_KEY")
fred = Fred(api_key=fred_api_key)

# Confirm connection by pulling one series
test_series = fred.get_series("UNRATE", observation_start="2020-01-01", observation_end="2020-06-30")
print("FRED connection confirmed. Sample unemployment values:")
print(test_series)

print("\n")
print("Used API key to call data and successfully downloaded a timeseries")
print("\n")

# -----------------------------------------------------------------------------
# Part 1: Load and clean presidential vote data
# -----------------------------------------------------------------------------
vote_data = pd.read_csv("demo/1976-2020-president.csv")

vote_data = vote_data[
    vote_data["party_detailed"].isin(["DEMOCRAT", "REPUBLICAN"])
].copy()

vote_data = (
    vote_data
    .groupby(["year", "candidate", "party_detailed"], as_index=False)
    .agg(
        candidatevotes=("candidatevotes", "sum"),
        totalvotes=("totalvotes", "sum")
    )
)

vote_data = vote_data[
    (~vote_data["candidate"].isin(["OTHER", ""])) &
    (vote_data["candidate"].notna())
].copy()

vote_data["vote_pct"] = vote_data["candidatevotes"] / vote_data["totalvotes"]
election_years = np.sort(vote_data["year"].unique())

# -----------------------------------------------------------------------------
# Part 2: Pull economic indicators from FRED (Q1/Q2 of election years)
# -----------------------------------------------------------------------------
obs_start = f"{int(election_years.min())}-01-01"
obs_end   = f"{int(election_years.max())}-06-30"

def pull_fred_quarterly(series_id, col_name, election_years, obs_start, obs_end):
    raw = fred.get_series(series_id, observation_start=obs_start, observation_end=obs_end)
    df = raw.to_frame(name=col_name)
    df.index = pd.to_datetime(df.index)
    df = df.resample("Q").mean().reset_index().rename(columns={"index": "date"})
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df = df[
        (df["year"].isin(election_years)) &
        (df["quarter"] <= 2)
    ][["year", "quarter", col_name]].copy()
    return df

unemployment_data = pull_fred_quarterly("UNRATE",    "unemployment_rate", election_years, obs_start, obs_end)
gdp_data          = pull_fred_quarterly("GDP",       "gdp",               election_years, obs_start, obs_end)
cpi_data          = pull_fred_quarterly("CPIAUCSL",  "cpi",               election_years, obs_start, obs_end)

# Task 1 extension: add two more FRED indicators
# Consumer Sentiment (UMCSENT) - captures voter mood beyond hard economic numbers
# Real Disposable Personal Income (DSPIC96) - measures how households actually feel economically
sentiment_data = pull_fred_quarterly("UMCSENT",  "consumer_sentiment", election_years, obs_start, obs_end)
income_data    = pull_fred_quarterly("DSPIC96",  "real_income",        election_years, obs_start, obs_end)

# Merge all indicators into one wide table (one row per year)
combined_long = (
    unemployment_data
    .merge(gdp_data,       on=["year", "quarter"], how="outer")
    .merge(cpi_data,       on=["year", "quarter"], how="outer")
    .merge(sentiment_data, on=["year", "quarter"], how="outer")
    .merge(income_data,    on=["year", "quarter"], how="outer")
    .sort_values(["year", "quarter"])
)

combined_wide = combined_long.pivot_table(
    index="year",
    columns="quarter",
    values=["unemployment_rate", "gdp", "cpi", "consumer_sentiment", "real_income"],
    aggfunc="first"
)
combined_wide.columns = [f"{var}_Q{q}" for var, q in combined_wide.columns]
combined_wide = combined_wide.reset_index()

# -----------------------------------------------------------------------------
# Part 3: Merge and build features
# -----------------------------------------------------------------------------
forecast_data = vote_data.merge(combined_wide, on="year", how="left").copy()

# Incumbent indicator (hard-coded, no if/else)
forecast_data["incumbent"] = 0
forecast_data.loc[(forecast_data["candidate"] == "FORD, GERALD")       & (forecast_data["year"] == 1976), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CARTER, JIMMY")      & (forecast_data["year"] == 1980), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "REAGAN, RONALD")     & (forecast_data["year"] == 1984), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE H.W.")  & (forecast_data["year"] == 1992), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CLINTON, BILL")      & (forecast_data["year"] == 1996), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE W.")    & (forecast_data["year"] == 2004), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "OBAMA, BARACK H.")   & (forecast_data["year"] == 2012), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "TRUMP, DONALD J.")   & (forecast_data["year"] == 2020), "incumbent"] = 1

# Q2 - Q1 change features (baseline features)
forecast_data["gdp_change"]       = forecast_data["gdp_Q2"]               - forecast_data["gdp_Q1"]
forecast_data["cpi_change"]       = forecast_data["cpi_Q2"]               - forecast_data["cpi_Q1"]
forecast_data["unemploy_change"]  = forecast_data["unemployment_rate_Q2"] - forecast_data["unemployment_rate_Q1"]

# Additional change features for extended model
forecast_data["sentiment_change"] = forecast_data["consumer_sentiment_Q2"] - forecast_data["consumer_sentiment_Q1"]
forecast_data["income_change"]    = forecast_data["real_income_Q2"]         - forecast_data["real_income_Q1"]

# Per-capita GDP approximation: log transform to handle scale
forecast_data["log_gdp_Q1"] = np.log(forecast_data["gdp_Q1"])

# Train/test split: train on years < 2020, test on 2020
forecast_data_training = forecast_data[forecast_data["year"] < 2020].copy()
forecast_data_testing  = forecast_data[forecast_data["year"] == 2020].copy()

# -----------------------------------------------------------------------------
# Task 2: Baseline model (replicate demo)
# -----------------------------------------------------------------------------
baseline_ols = smf.ols(
    "vote_pct ~ incumbent * unemploy_change + C(party_detailed) + year + I(year**2)",
    data=forecast_data_training
).fit()

forecast_data_testing["baseline_pred"] = baseline_ols.predict(forecast_data_testing)

baseline_mae  = mean_absolute_error(forecast_data_testing["vote_pct"], forecast_data_testing["baseline_pred"])
baseline_rmse = mean_squared_error(forecast_data_testing["vote_pct"],  forecast_data_testing["baseline_pred"]) ** 0.5

print("\n--- Baseline Model: 2020 Out-of-Sample ---")
print(forecast_data_testing[["candidate", "party_detailed", "vote_pct", "baseline_pred"]].to_string(index=False))
print(f"Baseline MAE:  {baseline_mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")

os.makedirs("problem_set/outputs/table", exist_ok=True)
os.makedirs("problem_set/outputs/figure", exist_ok=True)

forecast_data_testing[["candidate", "party_detailed", "vote_pct", "baseline_pred"]].to_csv("problem_set/outputs/table/baseline_2020_predictions.csv", index=False)
print("Saved to baseline_2020_predictions.csv")

print("\n")
print("Ran the baseline forecaster")
print("\n")

# -----------------------------------------------------------------------------
# Task 3: Improved model
# Improvements:
#   1) Add consumer sentiment change and real income change from FRED API
#   2) Add log(GDP) to capture nonlinear income effects
#   3) Add incumbent * sentiment_change interaction
# Out-of-sample design: hold out 2020 (train on years < 2020)
# -----------------------------------------------------------------------------
improved_ols = smf.ols(
    """vote_pct ~ incumbent * unemploy_change
                + incumbent * sentiment_change
                + income_change
                + log_gdp_Q1
                + C(party_detailed)
                + year + I(year**2)""",
    data=forecast_data_training
).fit()

forecast_data_testing["improved_pred"] = improved_ols.predict(forecast_data_testing)

improved_mae  = mean_absolute_error(forecast_data_testing["vote_pct"], forecast_data_testing["improved_pred"])
improved_rmse = mean_squared_error(forecast_data_testing["vote_pct"],  forecast_data_testing["improved_pred"]) ** 0.5

print("\n--- Improved Model: 2020 Out-of-Sample ---")
print(forecast_data_testing[["candidate", "party_detailed", "vote_pct", "improved_pred"]].to_string(index=False))
print(f"Improved MAE:  {improved_mae:.4f}")
print(f"Improved RMSE: {improved_rmse:.4f}")

forecast_data_testing[["candidate", "party_detailed", "vote_pct", "improved_pred"]].to_csv("problem_set/outputs/table/improved_2020_predictions.csv", index=False)
print("Saved to outputs/table/improved_2020_predictions.csv")

print("\n")
print("Implented model improvements, trained on before 2020 and tested on 2020, and reported out-of-sample performance")
print("\n")

# -----------------------------------------------------------------------------
# Task 4a: Comparison table (baseline vs improved)
# -----------------------------------------------------------------------------
comparison_table = pd.DataFrame({
    "Model":    ["Baseline OLS", "Improved OLS"],
    "Features": [
        "incumbent × unemploy_change, party, year²",
        "+ sentiment_change, income_change, log(GDP), incumbent × sentiment"
    ],
    "MAE":  [round(baseline_mae,  4), round(improved_mae,  4)],
    "RMSE": [round(baseline_rmse, 4), round(improved_rmse, 4)]
})

print("\n--- Model Comparison Table ---")
print(comparison_table.to_string(index=False))

comparison_table.to_csv("problem_set/outputs/table/model_comparison.csv", index=False)
print("Saved to problem_set/outputs/table/model_comparison.csv")

# -----------------------------------------------------------------------------
# Task 4b: Figure — predicted vs actual vote share for both models (2020)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

candidates = forecast_data_testing["candidate"].tolist()
actual      = forecast_data_testing["vote_pct"].tolist()
baseline_p  = forecast_data_testing["baseline_pred"].tolist()
improved_p  = forecast_data_testing["improved_pred"].tolist()

x = np.arange(len(candidates))
width = 0.25

ax.bar(x - width, actual,     width, label="Actual",        color="gray")
ax.bar(x,         baseline_p, width, label="Baseline OLS",  color="steelblue")
ax.bar(x + width, improved_p, width, label="Improved OLS",  color="darkorange")

ax.set_xticks(x)
ax.set_xticklabels([c.split(",")[0].title() for c in candidates], fontsize=10)
ax.set_ylabel("Vote Share")
ax.set_title("2020 Out-of-Sample: Actual vs Predicted Vote Share")
ax.legend()
ax.set_ylim(-1, 1)
plt.tight_layout()
plt.savefig("problem_set/outputs/figure/model_comparison.png", dpi=150)
print("\nFigure saved to model_comparison.png")

print("\n")
print("Comparison table and figure saved to directory")
print("\n")