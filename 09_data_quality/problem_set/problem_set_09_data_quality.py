###############################################################################
# pset_week11.py
# Week 11 Problem Set: Measurement Error and Placebo Tests
# Covers Q4, Q5, Q6 from the problem set
###############################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

np.random.seed(123)

os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------------------------------
# DGP (same as tutorial)
# -----------------------------------------------------------------------------
n = 5000

x_true = np.random.normal(loc=0.0, scale=1.0, size=n)
logit_p = 1.0 * x_true
p = 1.0 / (1.0 + np.exp(-logit_p))
d = np.random.binomial(n=1, p=p, size=n)

tau = 1.0
beta = 1.0
eps_y = np.random.normal(loc=0.0, scale=1.0, size=n)
y = tau * d + beta * x_true + eps_y

eps_pl = np.random.normal(loc=0.0, scale=1.0, size=n)
y_placebo = 0.0 * d + beta * x_true + eps_pl

df_base = pd.DataFrame({"y": y, "y_placebo": y_placebo, "d": d, "x_true": x_true})

print("Treatment rate:", df_base["d"].mean().round(3))

# -----------------------------------------------------------------------------
# Q4: Measurement error simulation across sigma_u values
# Reproduces tutorial results; outputs the table and figures needed for Q4
# -----------------------------------------------------------------------------
print("\n=== Q4: Measurement Error Simulation ===")

sigma_u_grid = [0.0, 0.2, 0.5, 1.0, 2.0]
R = 30

validation_share = 0.20
validation_idx = np.random.choice(np.arange(n), size=int(validation_share * n), replace=False)
is_validation = np.zeros(n, dtype=bool)
is_validation[validation_idx] = True

rows = []

for sigma_u in sigma_u_grid:
    tau_oracle_list, tau_naive_list, tau_cal_list = [], [], []
    beta_oracle_list, beta_naive_list, beta_cal_list = [], [], []

    for r in range(R):
        u = np.random.normal(loc=0.0, scale=sigma_u, size=n)
        x_obs = x_true + u

        df = df_base.copy()
        df["x_obs"] = x_obs

        fit_oracle = smf.ols("y ~ d + x_true", data=df).fit()
        tau_oracle_list.append(fit_oracle.params["d"])
        beta_oracle_list.append(fit_oracle.params["x_true"])

        fit_naive = smf.ols("y ~ d + x_obs", data=df).fit()
        tau_naive_list.append(fit_naive.params["d"])
        beta_naive_list.append(fit_naive.params["x_obs"])

        df_val = df.loc[is_validation, ["x_true", "x_obs"]].copy()
        fit_cal = smf.ols("x_true ~ x_obs", data=df_val).fit()
        df["x_hat"] = fit_cal.predict(df[["x_obs"]])
        fit_calibrated = smf.ols("y ~ d + x_hat", data=df).fit()
        tau_cal_list.append(fit_calibrated.params["d"])
        beta_cal_list.append(fit_calibrated.params["x_hat"])

    rows.append({
        "sigma_u": sigma_u,
        "tau_oracle_mean": round(float(np.mean(tau_oracle_list)), 4),
        "tau_naive_mean":  round(float(np.mean(tau_naive_list)), 4),
        "tau_cal_mean":    round(float(np.mean(tau_cal_list)), 4),
        "beta_oracle_mean": round(float(np.mean(beta_oracle_list)), 4),
        "beta_naive_mean":  round(float(np.mean(beta_naive_list)), 4),
        "beta_cal_mean":    round(float(np.mean(beta_cal_list)), 4),
    })
    print(f"  done sigma_u={sigma_u}")

results = pd.DataFrame(rows)
results.to_csv("outputs/measurement_error_results.csv", index=False)

# Q4 table: tau estimates
print("\n--- Q4 Table A: Treatment effect estimates (tau) ---")
print(results[["sigma_u", "tau_oracle_mean", "tau_naive_mean", "tau_cal_mean"]].to_string(index=False))

# Q4 table: beta estimates
print("\n--- Q4 Table B: Confounder coefficient estimates (beta) ---")
print(results[["sigma_u", "beta_oracle_mean", "beta_naive_mean", "beta_cal_mean"]].to_string(index=False))

# Q4 figure: tau vs sigma_u
plt.figure(figsize=(8, 5))
plt.plot(results["sigma_u"], results["tau_oracle_mean"], marker="o", label="Oracle")
plt.plot(results["sigma_u"], results["tau_naive_mean"], marker="o", label="Naive")
plt.plot(results["sigma_u"], results["tau_cal_mean"], marker="o", label="Calibration")
plt.axhline(tau, linestyle="--", label="True tau = 1.0")
plt.title("Q4: Treatment effect estimate vs measurement error (sigma_u)")
plt.xlabel("Measurement error SD (sigma_u)")
plt.ylabel("Estimated coefficient on d")
plt.legend()
plt.tight_layout()
plt.savefig("figures/q4_tau_vs_sigma.png", dpi=200)
plt.close()
print("Saved figures/q4_tau_vs_sigma.png")

# Q4 figure: beta attenuation vs sigma_u
plt.figure(figsize=(8, 5))
plt.plot(results["sigma_u"], results["beta_oracle_mean"], marker="o", label="Oracle")
plt.plot(results["sigma_u"], results["beta_naive_mean"], marker="o", label="Naive")
plt.plot(results["sigma_u"], results["beta_cal_mean"], marker="o", label="Calibration")
plt.axhline(beta, linestyle="--", label="True beta = 1.0")
plt.title("Q4: Confounder coefficient attenuation vs measurement error (sigma_u)")
plt.xlabel("Measurement error SD (sigma_u)")
plt.ylabel("Estimated coefficient on confounder")
plt.legend()
plt.tight_layout()
plt.savefig("figures/q4_beta_vs_sigma.png", dpi=200)
plt.close()
print("Saved figures/q4_beta_vs_sigma.png")

# -----------------------------------------------------------------------------
# Q5: Vary validation_share at fixed sigma_u = 1.0
# -----------------------------------------------------------------------------
print("\n=== Q5: Validation Share Sensitivity ===")

sigma_u_q5 = 1.0
validation_shares = [0.05, 0.20, 0.50]
q5_rows = []

for vs in validation_shares:
    val_idx = np.random.choice(np.arange(n), size=int(vs * n), replace=False)
    is_val = np.zeros(n, dtype=bool)
    is_val[val_idx] = True

    tau_naive_list_q5, tau_cal_list_q5 = [], []

    for r in range(R):
        u = np.random.normal(loc=0.0, scale=sigma_u_q5, size=n)
        x_obs = x_true + u

        df = df_base.copy()
        df["x_obs"] = x_obs

        fit_naive = smf.ols("y ~ d + x_obs", data=df).fit()
        tau_naive_list_q5.append(fit_naive.params["d"])

        df_val = df.loc[is_val, ["x_true", "x_obs"]].copy()
        fit_cal = smf.ols("x_true ~ x_obs", data=df_val).fit()
        df["x_hat"] = fit_cal.predict(df[["x_obs"]])
        fit_calibrated = smf.ols("y ~ d + x_hat", data=df).fit()
        tau_cal_list_q5.append(fit_calibrated.params["d"])

    q5_rows.append({
        "validation_share": vs,
        "tau_naive_mean": round(float(np.mean(tau_naive_list_q5)), 4),
        "tau_cal_mean":   round(float(np.mean(tau_cal_list_q5)), 4),
    })
    print(f"  done validation_share={vs}")

q5_results = pd.DataFrame(q5_rows)
q5_results.to_csv("outputs/q5_validation_share_results.csv", index=False)

print("\n--- Q5 Table: Calibration vs naive by validation share (sigma_u=1.0) ---")
print(q5_results.to_string(index=False))

# -----------------------------------------------------------------------------
# Q6: Outcome placebo + treatment permutation placebo
# -----------------------------------------------------------------------------
print("\n=== Q6: Placebo Tests ===")

sigma_u_q6 = 1.0
u_q6 = np.random.normal(loc=0.0, scale=sigma_u_q6, size=n)
x_obs_q6 = x_true + u_q6

df_q6 = df_base.copy()
df_q6["x_obs"] = x_obs_q6

# Outcome placebo: y_placebo ~ d + x_obs
# Coefficient on d should be near zero (d has no effect on y_placebo by construction)
fit_outcome_placebo = smf.ols("y_placebo ~ d + x_obs", data=df_q6).fit()
tau_outcome_placebo = float(fit_outcome_placebo.params["d"])
print(f"\nOutcome placebo: coef on d = {tau_outcome_placebo:.4f}  (expect ~0)")

# Treatment permutation placebo
fit_obs = smf.ols("y ~ d + x_obs", data=df_q6).fit()
tau_hat_obs = float(fit_obs.params["d"])
print(f"Observed tau_hat (naive, sigma_u=1.0): {tau_hat_obs:.4f}")

B = 500
tau_perm = []

for b in range(B):
    d_perm = np.random.permutation(df_q6["d"].values)
    df_b = df_q6.copy()
    df_b["d_perm"] = d_perm
    fit_b = smf.ols("y ~ d_perm + x_obs", data=df_b).fit()
    tau_perm.append(float(fit_b.params["d_perm"]))

tau_perm = np.array(tau_perm)
p_emp = (1.0 + np.sum(np.abs(tau_perm) >= np.abs(tau_hat_obs))) / (B + 1.0)

print(f"Empirical two-sided p-value: {p_emp:.4f}")

pd.DataFrame({"tau_perm": tau_perm}).to_csv("outputs/q6_permutation_distribution.csv", index=False)

# Q6 figure: permutation histogram
plt.figure(figsize=(8, 5))
plt.hist(tau_perm, bins=30, alpha=0.8, label="Permutation distribution")
plt.axvline(tau_hat_obs, linestyle="--", linewidth=2, color="red",
            label=f"Observed tau_hat = {tau_hat_obs:.3f}")
plt.axvline(-tau_hat_obs, linestyle="--", linewidth=1, color="red")
plt.title(f"Q6: Treatment permutation placebo (sigma_u={sigma_u_q6})\nEmpirical p-value = {p_emp:.3f}")
plt.xlabel("Coefficient on permuted treatment")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("figures/q6_permutation_placebo_hist.png", dpi=200)
plt.close()
print("Saved figures/q6_permutation_placebo_hist.png")

# -----------------------------------------------------------------------------
# Summary of key numbers for write-up
# -----------------------------------------------------------------------------
print("\n=== Summary for write-up ===")
print(f"Q6 outcome placebo coef on d:     {tau_outcome_placebo:.4f}")
print(f"Q6 observed tau_hat (naive):       {tau_hat_obs:.4f}")
print(f"Q6 empirical p-value (two-sided):  {p_emp:.4f}")
print("\nQ5 results:")
print(q5_results.to_string(index=False))
print("\nDone.")
