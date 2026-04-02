###############################################################################
# SoDA 501 - Week 11: Databases and SQL
# Problem Set Solution
###############################################################################

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from datetime import date, timedelta

os.makedirs("outputs/figure", exist_ok=True)
os.makedirs("outputs/table", exist_ok=True)

# -----------------------------------------------------------------------------
# Build the database (from demo script)
# -----------------------------------------------------------------------------

con = sqlite3.connect("campaign_finance.db")
cur = con.cursor()

cur.execute("DROP TABLE IF EXISTS contributions;")
cur.execute("DROP TABLE IF EXISTS contributors;")
cur.execute("DROP TABLE IF EXISTS candidates;")
con.commit()

cur.execute("""
  CREATE TABLE candidates (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    party TEXT,
    office TEXT,
    winner INTEGER
  );
""")

cur.execute("""
  CREATE TABLE contributors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    occupation TEXT,
    employer TEXT,
    state TEXT
  );
""")

cur.execute("""
  CREATE TABLE contributions (
    id INTEGER PRIMARY KEY,
    contributor_id INTEGER,
    candidate_id INTEGER,
    amount REAL,
    date TEXT,
    FOREIGN KEY (contributor_id) REFERENCES contributors(id),
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
  );
""")
con.commit()

np.random.seed(123)

candidate_ids = np.arange(1, 101)
candidate_names = np.array([f"Candidate {i}" for i in candidate_ids])
candidate_parties = np.random.choice(
    ["Democrat", "Republican", "Independent"],
    size=100, replace=True, p=[0.45, 0.45, 0.10]
)
candidate_offices = np.random.choice(
    ["Senate", "House", "Governor", "State Senate", "State House"],
    size=100, replace=True
)
candidate_winner = np.random.choice([1, 0], size=100, replace=True, p=[0.5, 0.5])

candidates = pd.DataFrame({
    "id": candidate_ids,
    "name": candidate_names,
    "party": candidate_parties,
    "office": candidate_offices,
    "winner": candidate_winner
})

contributor_ids = np.arange(1, 100001)
contributor_names = np.array([f"Contributor {i}" for i in contributor_ids])
contributor_occupations = np.random.choice(
    ["Engineer", "Teacher", "Doctor", "Lawyer", "Business Owner"],
    size=100000, replace=True
)
contributor_employers = np.array([f"Company {i}" for i in np.random.randint(1, 5001, size=100000)])
state_abb = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]
contributor_states = np.random.choice(state_abb, size=100000, replace=True)

contributors = pd.DataFrame({
    "id": contributor_ids,
    "name": contributor_names,
    "occupation": contributor_occupations,
    "employer": contributor_employers,
    "state": contributor_states
})

contribution_ids = np.arange(1, 1000001)
contribution_contributor_ids = np.random.randint(1, 100001, size=1000000)
contribution_candidate_ids = np.random.randint(1, 101, size=1000000)
contribution_amounts = np.round(
    np.random.lognormal(mean=np.log(1000), sigma=1, size=1000000), 2
)
start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)
n_days = (end_date - start_date).days + 1
random_day_offsets = np.random.randint(0, n_days, size=1000000)
contribution_dates = np.array(
    [(start_date + timedelta(days=int(d))).isoformat() for d in random_day_offsets]
)

contributions = pd.DataFrame({
    "id": contribution_ids,
    "contributor_id": contribution_contributor_ids,
    "candidate_id": contribution_candidate_ids,
    "amount": contribution_amounts,
    "date": contribution_dates
})

candidates.to_sql("candidates", con, if_exists="append", index=False, chunksize=5000)
contributors.to_sql("contributors", con, if_exists="append", index=False, chunksize=5000)
contributions.to_sql("contributions", con, if_exists="append", index=False, chunksize=5000)
con.commit()

cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_contributor_id ON contributions (contributor_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_candidate_id   ON contributions (candidate_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_amount         ON contributions (amount);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_date           ON contributions (date);")
con.commit()

# -----------------------------------------------------------------------------
# Task 1: Row counts and schema inspection
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("TASK 1: ROW COUNTS AND SCHEMA")
print("="*60)

for table in ["candidates", "contributors", "contributions"]:
    count = pd.read_sql_query(f"SELECT COUNT(*) AS row_count FROM {table};", con)
    print(f"\nRow count -- {table}:")
    print(count.to_string(index=False))

print("\n--- Schema: candidates ---")
schema_candidates = pd.read_sql_query("PRAGMA table_info(candidates);", con)
print(schema_candidates.to_string(index=False))
schema_candidates.to_csv("outputs/table/schema_candidates.csv", index=False)

print("\n--- Schema: contributors ---")
schema_contributors = pd.read_sql_query("PRAGMA table_info(contributors);", con)
print(schema_contributors.to_string(index=False))
schema_contributors.to_csv("outputs/table/schema_contributors.csv", index=False)

print("\n--- Schema: contributions ---")
schema_contributions = pd.read_sql_query("PRAGMA table_info(contributions);", con)
print(schema_contributions.to_string(index=False))
schema_contributions.to_csv("outputs/table/schema_contributions.csv", index=False)

# -----------------------------------------------------------------------------
# Task 2: Join + aggregation -- total contributions by party (amount > 1000)
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("TASK 2: JOINS AND AGGREGATION")
print("="*60)

query_party = """
  SELECT
    ca.party,
    SUM(co.amount)   AS total_amount,
    COUNT(co.id)     AS num_contributions
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  WHERE co.amount > 1000
  GROUP BY ca.party
  ORDER BY total_amount DESC;
"""

party_totals = pd.read_sql_query(query_party, con)

print("\nTotal contributions by party (amount > $1000):")
print(party_totals.to_string(index=False))
party_totals.to_csv("outputs/table/party_totals.csv", index=False)

plt.figure(figsize=(7, 5))
plt.bar(party_totals["party"], party_totals["total_amount"], color=["#2166ac", "#d6604d", "#4dac26"])
plt.title("Total Contributions by Party (Amount > $1,000)")
plt.xlabel("Party")
plt.ylabel("Total Amount ($)")
plt.tight_layout()
plt.savefig("outputs/figure/party_contributions.png", dpi=150)
plt.close()
print("\nSaved figure: outputs/figure/party_contributions.png")

# -----------------------------------------------------------------------------
# Task 3: Indexes and query plan
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("TASK 3: INDEXES AND QUERY PLAN")
print("="*60)

print("\nIndexes on contributions table:")
indexes = pd.read_sql_query("""
  SELECT name, sql
  FROM sqlite_master
  WHERE type = 'index'
    AND tbl_name = 'contributions';
""", con)
print(indexes.to_string(index=False))
indexes.to_csv("outputs/table/indexes.csv", index=False)

print("\nEXPLAIN QUERY PLAN -- filter by amount:")
explain_query = """
  EXPLAIN QUERY PLAN
  SELECT co.id, co.amount, ca.party
  FROM contributions co
  JOIN candidates ca ON co.candidate_id = ca.id
  WHERE co.amount > 1000;
"""
plan = pd.read_sql_query(explain_query, con)
print(plan.to_string(index=False))
plan.to_csv("outputs/table/query_plan.csv", index=False)

con.close()
