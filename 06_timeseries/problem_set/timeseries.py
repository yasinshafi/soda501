##############################
####### PROBLEM SET ##########
####### TIME SERIES ##########
##############################

##############################
### CODING DEMO SETUP ########
##############################

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- 0) Setup
np.random.seed(123)

# --- 1) Create a synthetic daily time series (trend + weekly seasonality + AR(1) noise)
n = 600
dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
t = np.arange(1, n + 1)

trend = 0.02 * t
weekly = 1.2 * np.sin(2 * np.pi * t / 7)

phi = 0.65
eps = np.random.normal(loc=0.0, scale=1.0, size=n)
ar_noise = np.empty(n)
ar_noise[0] = eps[0]
for i in range(1, n):
  ar_noise[i] = phi * ar_noise[i - 1] + eps[i]

y = 10 + trend + weekly + ar_noise

df = pd.DataFrame({"date": dates, "t": t, "y": y})

# --- 2) Visualize the series
plt.figure()
plt.plot(df["date"], df["y"])
plt.title("Synthetic daily time series: trend + weekly seasonality + AR(1) noise")
plt.xlabel("Date")
plt.ylabel("y")
plt.tight_layout()
plt.show()

############################################
# PART A: Time leakage demo (random split vs time split)
############################################

# --- 3) WRONG evaluation: random train/test split (time leakage)
np.random.seed(123)
test_frac = 0.20
test_n = int(np.floor(n * test_frac))

all_idx = np.arange(n)
test_idx_random = np.random.choice(all_idx, size=test_n, replace=False)
train_idx_random = np.setdiff1d(all_idx, test_idx_random)

y_train_random = df.loc[train_idx_random, "y"].to_numpy()
y_test_random = df.loc[test_idx_random, "y"].to_numpy()

# Fit ARIMA(1,0,0) on randomly selected training points (conceptually wrong for time series)
fit_random = ARIMA(y_train_random, order=(1, 0, 0)).fit()
pred_random = fit_random.forecast(steps=len(y_test_random))

rmse_random = np.sqrt(np.mean((y_test_random - pred_random) ** 2))

print("\n==============================")
print(f"WRONG: Random split RMSE (time leakage): {rmse_random:.6f}")
print("==============================")

# --- 4) RIGHT evaluation: train on past, test on future
cut = n - test_n
train_idx_time = np.arange(0, cut)
test_idx_time = np.arange(cut, n)

y_train_time = df.loc[train_idx_time, "y"].to_numpy()
y_test_time = df.loc[test_idx_time, "y"].to_numpy()

fit_time = ARIMA(y_train_time, order=(1, 0, 0)).fit()
pred_time = fit_time.forecast(steps=len(y_test_time))

rmse_time = np.sqrt(np.mean((y_test_time - pred_time) ** 2))

print("\n==============================")
print(f"RIGHT: Time split RMSE (train past, test future): {rmse_time:.6f}")
print("==============================")

# --- 5) Plot the correct evaluation (train vs test and forecast)
plt.figure()
plt.plot(df["date"], df["y"], label="Observed y")
plt.axvline(df.loc[cut, "date"], linestyle="--", label="Train/Test cutoff")
plt.plot(df.loc[test_idx_time, "date"], pred_time, label="Forecast (future)")
plt.title("Correct evaluation: train on past, test on future")
plt.xlabel("Date")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

############################################
# PART B: Synthetic DGP demo (autocorrelation + trend) + ACF/PACF diagnostics
############################################

# --- 6) Generate data from a known DGP: trend + AR(1) errors
# DGP: y_t = alpha + delta*t + e_t ; e_t = phi*e_{t-1} + u_t
np.random.seed(123)
n2 = 300
t2 = np.arange(1, n2 + 1)

alpha = 5
delta = 0.03
phi2 = 0.75
u = np.random.normal(loc=0.0, scale=1.0, size=n2)

e = np.empty(n2)
e[0] = u[0]
for i in range(1, n2):
  e[i] = phi2 * e[i - 1] + u[i]

y2 = alpha + delta * t2 + e

# --- 7) Plot the DGP series
plt.figure()
plt.plot(t2, y2)
plt.title("Synthetic DGP: linear trend + AR(1) errors")
plt.xlabel("t")
plt.ylabel("y_t")
plt.tight_layout()
plt.show()

# --- 8) Diagnose dependence with ACF and PACF
plt.figure()
plot_acf(y2, ax=plt.gca(), lags=40)
plt.title("ACF of y_t (trend + AR errors)")
plt.tight_layout()
plt.show()

plt.figure()
plot_pacf(y2, ax=plt.gca(), lags=40, method="ywm")
plt.title("PACF of y_t (trend + AR errors)")
plt.tight_layout()
plt.show()

# --- 9) Detrend and re-check ACF/PACF on residuals
# Remove linear trend via OLS: y2 ~ 1 + t2
X = np.column_stack([np.ones(n2), t2])
beta_hat = np.linalg.lstsq(X, y2, rcond=None)[0]
y2_hat = X @ beta_hat
resid2 = y2 - y2_hat

plt.figure()
plt.plot(t2, resid2)
plt.title("Residuals after removing linear trend (should still show AR structure)")
plt.xlabel("t")
plt.ylabel("residual")
plt.tight_layout()
plt.show()

plt.figure()
plot_acf(resid2, ax=plt.gca(), lags=40)
plt.title("ACF of residuals (trend removed)")
plt.tight_layout()
plt.show()

plt.figure()
plot_pacf(resid2, ax=plt.gca(), lags=40, method="ywm")
plt.title("PACF of residuals (trend removed)")
plt.tight_layout()
plt.show()

# --- 10) Fit an AR(1) model to residuals and compare estimated phi to truth
fit_ar1 = ARIMA(resid2, order=(1, 0, 0)).fit()
phi_hat = fit_ar1.params[1]  # AR1 coefficient (params[0] is intercept by default)

print("\n==============================")
print(f"DGP truth phi2 = {phi2}")
print(f"Estimated AR(1) phi from residuals = {phi_hat:.6f}")
print("==============================")

# --- 11) Narration-ready takeaway
print("\nNarration-ready takeaway:")
print("- In the DGP, we *know* the errors are AR(1), so observations are dependent over time.")
print("- ACF/PACF make that dependence visible.")
print("- Removing trend helps isolate autocorrelation in the error process.")
print("- Separately: random splits leak time and look too good; past->future splits are the honest default.")

############################################
########### PROBLEM SET TASKS ##############
############################################

############################################
# TASK 1: Decomposition (trend + seasonality + residual)
############################################

from statsmodels.tsa.seasonal import STL

# Convert to a pandas Series with DatetimeIndex (required for STL)
y_series = pd.Series(df["y"].values, index=pd.DatetimeIndex(df["date"]))

# Run STL decomposition with weekly frequency (period=7)
stl = STL(y_series, period=7, robust=True)
stl_result = stl.fit()

# Extract components
observed   = stl_result.observed
trend_comp = stl_result.trend
seasonal_comp = stl_result.seasonal
resid_comp = stl_result.resid

# Plot all four panels in one figure
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

axes[0].plot(df["date"], observed, lw=0.8)
axes[0].set_ylabel("Observed")

axes[1].plot(df["date"], trend_comp, lw=0.8, color="orange")
axes[1].set_ylabel("Trend")

axes[2].plot(df["date"], seasonal_comp, lw=0.8, color="green")
axes[2].set_ylabel("Seasonality")

axes[3].plot(df["date"], resid_comp, lw=0.8, color="red")
axes[3].set_ylabel("Residual")
axes[3].set_xlabel("Date")

fig.suptitle("STL Decomposition: trend + weekly seasonality + residual", y=1.01)
plt.tight_layout()

os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/figures/decomposition.png")


############################################
# TASK 2: Rolling-origin backtest (h=1 step ahead)
############################################

init_window = 300   # first 300 days as initial training window
h = 1               # forecast horizon

dates_arr = df["date"].to_numpy()
y_arr     = df["y"].to_numpy()

backtest_dates = []
backtest_y     = []
backtest_yhat  = []
backtest_error = []

for t in range(init_window, n - h):
    y_train_bt = y_arr[:t]
    y_true     = y_arr[t]                          # one step ahead true value

    fit_bt  = ARIMA(y_train_bt, order=(1, 0, 0)).fit()
    y_hat   = fit_bt.forecast(steps=h)[0]          # scalar forecast

    error = y_true - y_hat

    backtest_dates.append(dates_arr[t])
    backtest_y.append(y_true)
    backtest_yhat.append(y_hat)
    backtest_error.append(error)

# Compute backtest RMSE
backtest_errors_arr = np.array(backtest_error)
rmse_backtest = np.sqrt(np.mean(backtest_errors_arr ** 2))

print("\n==============================")
print(f"Rolling-origin backtest RMSE : {rmse_backtest:.6f}")
print(f"Single time-split RMSE       : {rmse_time:.6f}")
print("==============================")

# Save backtest errors CSV
os.makedirs("outputs/tables", exist_ok=True)
df_backtest = pd.DataFrame({
    "date" : backtest_dates,
    "y"    : backtest_y,
    "yhat" : backtest_yhat,
    "error": backtest_error
})
df_backtest.to_csv("outputs/tables/backtest_errors.csv", index=False)
print("Saved: outputs/tables/backtest_errors.csv")

# Plot y vs yhat over the test region
plt.figure(figsize=(10, 4))
plt.plot(df_backtest["date"], df_backtest["y"],    label="Observed y",    lw=0.8)
plt.plot(df_backtest["date"], df_backtest["yhat"], label="Forecast yhat", lw=0.8, linestyle="--", color="red")
plt.title("Rolling-origin backtest: observed vs one-step-ahead forecast")
plt.xlabel("Date")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/backtest_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/figures/backtest_forecast.png")