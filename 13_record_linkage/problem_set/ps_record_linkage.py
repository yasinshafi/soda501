###############################################################################
# SoDA 501 – Record Linkage Problem Set (Python translation)
# Questions 4, 5, 6
###############################################################################

# pip install pandas numpy matplotlib recordlinkage rapidfuzz

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import recordlinkage as rl
from rapidfuzz.distance import Levenshtein

os.makedirs("outputs/figure", exist_ok=True)
os.makedirs("outputs/table",  exist_ok=True)

# =============================================================================
# PART 0 – Generate synthetic data (identical to instructor script)
# =============================================================================
np.random.seed(123)
n = 10000

first_names = ["John", "Jane", "Michael", "Emily", "David",
               "Sarah", "William", "Emma", "James", "Olivia"]
last_names  = ["Smith", "Johnson", "Williams", "Brown", "Jones",
               "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

df_a = pd.DataFrame({
    "id":        np.arange(1, n + 1),
    "firstname": np.random.choice(first_names, size=n, replace=True),
    "lastname":  np.random.choice(last_names,  size=n, replace=True),
    "birthyear": np.random.randint(1970, 2001,  size=n),
    "zipcode":   np.random.randint(10000, 20001, size=n),
})
df_a = df_a.drop_duplicates().reset_index(drop=True)

df_b = df_a.copy()

mod_firstname = np.random.rand(len(df_b)) < 0.25
mod_lastname  = np.random.rand(len(df_b)) < 0.25
mod_birthyear = np.random.rand(len(df_b)) < 0.25

for i in np.where(mod_firstname)[0]:
    chars = list(df_b.loc[i, "firstname"])
    positions = np.random.choice(len(chars),
                                 np.random.randint(1, len(chars) + 1),
                                 replace=False)
    for pos in positions:
        chars[pos] = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"))
    df_b.loc[i, "firstname"] = "".join(chars)

for i in np.where(mod_lastname)[0]:
    chars = list(df_b.loc[i, "lastname"])
    positions = np.random.choice(len(chars),
                                 np.random.randint(1, len(chars) + 1),
                                 replace=False)
    for pos in positions:
        chars[pos] = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"))
    df_b.loc[i, "lastname"] = "".join(chars)

idx_by = np.where(mod_birthyear)[0]
df_b.loc[idx_by, "birthyear"] = (
    df_b.loc[idx_by, "birthyear"].to_numpy()
    + np.random.choice(np.arange(-2, 3), size=len(idx_by), replace=True)
)

df_a.to_csv("dataset_a.csv", index=False)
df_b.to_csv("dataset_b.csv", index=False)

# =============================================================================
# QUESTION 4 – Deterministic matching
# =============================================================================
df_a = pd.read_csv("dataset_a.csv").set_index("id")
df_b = pd.read_csv("dataset_b.csv").set_index("id")

det_matches = (
    df_a.reset_index()
        .merge(df_b.reset_index(),
               on=["firstname", "lastname", "birthyear", "zipcode"],
               how="inner",
               suffixes=(".a", ".b"))
)

n_det      = det_matches.shape[0]
match_rate = n_det / len(df_a)

print("=" * 60)
print("QUESTION 4 – Deterministic matching")
print(f"  Deterministic matches : {n_det}")
print(f"  Match rate            : {match_rate:.4f}  ({match_rate*100:.2f}%)")

# Save summary to table
q4_summary = pd.DataFrame({
    "n_df_a":         [len(df_a)],
    "det_matches":    [n_det],
    "match_rate":     [round(match_rate, 4)],
})
q4_summary.to_csv("outputs/table/q4_deterministic_summary.csv", index=False)


# =============================================================================
# QUESTION 5 – Probabilistic matching + threshold curve
# =============================================================================

# --- 5A: Blocking on zipcode ---
indexer = rl.Index()
indexer.block("zipcode")
candidate_pairs = indexer.index(df_a, df_b)
print("=" * 60)
print(f"QUESTION 5 – Candidate pairs after blocking: {len(candidate_pairs):,}")

# --- 5B: Feature construction ---
compare = rl.Compare()
compare.string("firstname", "firstname", method="jarowinkler",
               threshold=0.85, label="firstname_sim")
compare.string("lastname",  "lastname",  method="jarowinkler",
               threshold=0.85, label="lastname_sim")
compare.numeric("birthyear", "birthyear", method="gauss",
                offset=0, scale=2, label="birthyear_sim")
compare.exact("zipcode", "zipcode", label="zipcode_exact")

features = compare.compute(candidate_pairs, df_a, df_b)
features["birthyear_sim"] = (features["birthyear_sim"] >= 0.5).astype(int)

# --- 5C: Fit ECM model ---
ecm = rl.ECMClassifier()
ecm.fit(features)
posterior = ecm.prob(features)

posterior_df = posterior.reset_index()
posterior_df.columns = ["id_a", "id_b", "posterior"]

# --- 5D: Threshold grid → match counts ---
threshold_grid = np.arange(0, 1.01, 0.01)
match_counts   = [(posterior >= th).sum() for th in threshold_grid]

count_df = pd.DataFrame({"threshold": threshold_grid, "matches": match_counts})
count_df.to_csv("outputs/table/q5_threshold_match_counts.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(count_df["threshold"], count_df["matches"], color="#2563eb", linewidth=2)
ax.axvline(0.5,  color="gray",   linestyle="--", linewidth=1, label="t = 0.50")
ax.axvline(0.85, color="#dc2626", linestyle="--", linewidth=1, label="t = 0.85")
ax.set_xlabel("Threshold (posterior probability cutoff)")
ax.set_ylabel("Number of matches")
ax.set_title("Q5 – Match count vs. threshold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend()
fig.tight_layout()
fig.savefig("outputs/figure/q5_threshold_curve.png", dpi=150)
plt.close(fig)
print("  Figure saved → outputs/figure/q5_threshold_curve.png")


# =============================================================================
# QUESTION 6 – Match quality diagnostics + threshold justification
# =============================================================================

# --- 6A: Low-threshold candidate set + posterior bins ---
matches_low = posterior_df[posterior_df["posterior"] > 1e-6].copy()

bin_edges  = np.arange(0.0, 1.0000001, 0.1)
bin_labels = [
    "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
    "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"
]
matches_low["threshold_bin"] = pd.cut(
    matches_low["posterior"], bins=bin_edges, labels=bin_labels, right=True
)

# --- 6B: Join back to df_a, df_b to get field values ---
pairs = (
    matches_low
    .merge(df_a.reset_index(), left_on="id_a", right_on="id", how="left")
    .drop(columns=["id"])
    .rename(columns={"firstname": "firstname_a", "lastname": "lastname_a",
                     "birthyear": "birthyear_a", "zipcode": "zipcode_a"})
)
pairs = (
    pairs
    .merge(df_b.reset_index(), left_on="id_b", right_on="id", how="left")
    .drop(columns=["id"])
    .rename(columns={"firstname": "firstname_b", "lastname": "lastname_b",
                     "birthyear": "birthyear_b", "zipcode": "zipcode_b"})
)

# --- 6C: String and numeric distances ---
pairs["first_lv"]   = [Levenshtein.distance(a, b)
                       for a, b in zip(pairs["firstname_a"], pairs["firstname_b"])]
pairs["last_lv"]    = [Levenshtein.distance(a, b)
                       for a, b in zip(pairs["lastname_a"],  pairs["lastname_b"])]
pairs["birth_diff"] = (pairs["birthyear_a"] - pairs["birthyear_b"]).abs()

# --- Aggregate by bin ---
avg_by_bin = (
    pairs.groupby("threshold_bin", observed=False)
         .agg(mean_first=("first_lv",   "mean"),
              mean_last =("last_lv",    "mean"),
              mean_birth=("birth_diff", "mean"),
              n         =("posterior",  "size"))
         .reset_index()
)
avg_by_bin.to_csv("outputs/table/q6_quality_by_bin.csv", index=False)
print("=" * 60)
print("QUESTION 6 – Mean distances by posterior bin")
print(avg_by_bin.to_string(index=False))

# --- 6D: Plot – boxplots of first-name Levenshtein distance by bin ---
# (use a subsample for boxplot performance on large data)
sample = pairs.sample(min(30000, len(pairs)), random_state=42)
order  = bin_labels  # ensures correct x-axis ordering

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col, ylabel, title in zip(
    axes,
    ["first_lv",   "last_lv",    "birth_diff"],
    ["Levenshtein distance\n(first name)",
     "Levenshtein distance\n(last name)",
     "Absolute birth year\ndifference"],
    ["First name distance by posterior bin",
     "Last name distance by posterior bin",
     "Birth year difference by posterior bin"]
):
    groups = [sample.loc[sample["threshold_bin"] == b, col].dropna().values
              for b in order]
    ax.boxplot(groups, labels=order, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#bfdbfe"),
               medianprops=dict(color="#1d4ed8", linewidth=2))
    ax.set_xlabel("Posterior probability bin")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Q6 – Match quality vs. posterior probability", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("outputs/figure/q6_quality_boxplots.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Figure saved → outputs/figure/q6_quality_boxplots.png")

# --- Also save line-plot version of mean distances ---
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(avg_by_bin["threshold_bin"].astype(str), avg_by_bin["mean_first"],
        marker="o", label="First name (LV)", color="#2563eb")
ax.plot(avg_by_bin["threshold_bin"].astype(str), avg_by_bin["mean_last"],
        marker="s", label="Last name (LV)",  color="#16a34a")
ax.plot(avg_by_bin["threshold_bin"].astype(str), avg_by_bin["mean_birth"],
        marker="^", label="Birth year (abs diff)", color="#dc2626")
ax.set_xlabel("Posterior probability bin")
ax.set_ylabel("Mean distance")
ax.set_title("Q6 – Mean field distances by posterior bin")
ax.tick_params(axis="x", rotation=45)
ax.legend()
fig.tight_layout()
fig.savefig("outputs/figure/q6_mean_distance_lines.png", dpi=150)
plt.close(fig)
print("  Figure saved → outputs/figure/q6_mean_distance_lines.png")

# --- Chosen threshold ---
CHOSEN_THRESHOLD = 0.85
final_matches = posterior_df[posterior_df["posterior"] >= CHOSEN_THRESHOLD]
n_prob = final_matches.shape[0]

print()
print(f"  Probabilistic matches at t = {CHOSEN_THRESHOLD}: {n_prob:,}")
print(f"  Deterministic matches                          : {n_det:,}")
