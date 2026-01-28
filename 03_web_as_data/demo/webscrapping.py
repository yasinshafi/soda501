###############################################################################
# Web Scraping + Google Scholar Tutorial: Python (Penn State Faculty Example)
# Author: Jared Edgerton
# Date: (fill in manually or use datetime.date.today())
#
# This script demonstrates:
#   1) Web scraping a Wikipedia infobox table (warm-up example)
#   2) Web scraping Penn State faculty pages (text + targeted HTML scraping)
#   3) Pulling citation metrics from Google Scholar (via the scholarly package)
#   4) Simple plotting with matplotlib
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - You will see the same steps repeated for each professor so students can
#   follow the logic and edit one piece at a time.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.
#
# In a terminal:
#   pip install requests lxml cssselect pandas matplotlib scholarly
#
# Notes:
# - lxml + cssselect lets us use CSS selectors (like "table.infobox") and XPath.
# - Google Scholar scraping can be brittle (rate limits / CAPTCHAs). Consider
#   pre-running tonight and saving outputs if you want a guaranteed live demo.

import re
import requests
import pandas as pd
import matplotlib.pyplot as plt

from lxml import html
from scholarly import scholarly

# Polite headers (good practice for scraping)
HEADERS = {
    "User-Agent": "psu-webscrape-class-demo/1.0 (contact: you@example.com)"
}

# -----------------------------------------------------------------------------
# Part 1: Web Scraping (Wikipedia Warm-up + Penn State Faculty Pages)
# -----------------------------------------------------------------------------
# We will do web scraping in two stages:
#
# A) Wikipedia warm-up (table scraping)
# - Read a Wikipedia page
# - Extract the "infobox" table
# - Clean it into a Key/Value table
#
# B) Penn State faculty pages (text + targeted HTML scraping)
# - Read each faculty member’s PSU profile page
# - Pull the page text and extract:
#     * A title line (regex)
#     * A PSU email address (regex)
# - Pull structured items like:
#     * "Areas of Interest" (HTML nodes)
#     * "Research Interests" (HTML nodes)
#
# Note:
# - Department websites do not always have the same structure.
# - Some pages may have "Areas of Interest" while others have "Research Interests".

# -----------------------------------------------------------------------------
# Part 1A: Wikipedia Warm-up (Scraping an Infobox Table)
# -----------------------------------------------------------------------------
# In Wikipedia, many biography pages include an "infobox" on the right side.
# That infobox is typically stored as an HTML table with class "infobox".

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/Donald_Trump"

# Read the HTML content
resp = requests.get(url, headers=HEADERS)
page = html.fromstring(resp.content)

###############################################################################
# 2) Identify the HTML “thing” you want
#
# Wikipedia biographies commonly have an "infobox": a table on the right side
# that summarizes key facts (Born, Education, Occupation, etc.).
#
# In the HTML, this is usually a <table> element with class="infobox".
#
# CSS selector reminder:
#   - "table.infobox" means: a <table> tag with class="infobox"
#   - "." indicates a class; "#" indicates an id
###############################################################################

# Extract the infobox table (usually the first one on the page)
infobox_nodes = page.cssselect("table.infobox")

# (Optional safety check for teaching: if none found, you’d stop or change strategy)
# len(infobox_nodes)

infobox_node = infobox_nodes[0]

###############################################################################
# 3) Convert the HTML element to a data structure
#
# pandas.read_html() converts an HTML <table> into a pandas DataFrame.
# Wikipedia infobox tables often have merged cells / uneven rows; pandas handles
# most of this automatically when reading the HTML table.
###############################################################################
infobox_html = html.tostring(infobox_node, encoding="unicode")
infobox_raw = pd.read_html(infobox_html)[0]

###############################################################################
# 4) Standardize column names (so cleaning code is consistent)
#
# Infobox tables usually come out as 2 columns:
#   left column  = "label" (e.g., "Born")
#   right column = "value" (e.g., "January 1, 19xx ...")
#
# But sometimes rows span columns, and the parsed table can look messy.
###############################################################################
infobox_raw.columns = [f"X{i}" for i in range(1, infobox_raw.shape[1] + 1)]

###############################################################################
# 5) Clean into "key-value" (tidy) format
#
# Goal: one row per field, with:
#   field = label text
#   value = value text
#
# Notes:
# - Filter removes rows that aren’t label/value
# - Whitespace “squish” collapses repeated whitespace + trims ends
###############################################################################
infobox_kv = (
    infobox_raw.loc[
        infobox_raw["X1"].notna()
        & infobox_raw["X2"].notna()
        & (infobox_raw["X1"].astype(str).str.strip() != "")
        & (infobox_raw["X2"].astype(str).str.strip() != ""),
        ["X1", "X2"],
    ]
    .rename(columns={"X1": "field", "X2": "value"})
)

infobox_kv["field"] = (
    infobox_kv["field"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

infobox_kv["value"] = (
    infobox_kv["value"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# In a notebook, you could just write: infobox_kv
print(infobox_kv.head(25))

# -----------------------------------------------------------------------------
# Part 1B: Hard-code three Penn State faculty (social sciences broadly)
# -----------------------------------------------------------------------------
# These are the three faculty members we will use throughout the script.
# (We will repeat the same scraping steps for each person.)

matt_name = "Matt Golder"
matt_dept = "Political Science (College of the Liberal Arts)"
matt_url  = "https://polisci.la.psu.edu/people/mrg19/"

sona_name = "Sona N. Golder"
sona_dept = "Political Science (College of the Liberal Arts)"
sona_url  = "https://polisci.la.psu.edu/people/sng11/"

derek_name = "Derek Kreager"
derek_dept = "Sociology & Criminology (College of the Liberal Arts)"
derek_url  = "https://sociology.la.psu.edu/people/derek-kreager/"

# -----------------------------------------------------------------------------
# Step 1: Scrape Matt Golder (one complete example, step-by-step)
# -----------------------------------------------------------------------------
# 1) Read the PSU profile page
matt_resp = requests.get(matt_url, headers=HEADERS)
matt_page = html.fromstring(matt_resp.content)

# HTML headings run from <h1> through <h6> (six levels total), and they indicate document structure:
#   h1 = page title (top-level; usually one per page)
#   h2 = major section
#   h3 = subsection
#   h4 = sub-subsection
#   h5 = very fine-grained subsection (rare in practice)
#   h6 = smallest heading level (very rare)
# In practice, most modern pages rarely use h5/h6; you usually see h1–h3 (sometimes h4).
# For scraping, the *heading text* ("Research", "Publications", etc.) often matters more than the level,
# so it’s common to search across multiple heading levels when mapping the page structure.

heads = [h.text_content().strip() for h in matt_page.cssselect("h1, h2, h3, h4, h5, h6")]
print(heads)

print(len(matt_page.cssselect("main, article, header, nav, footer")))
print(len(matt_page.cssselect("p")))
print(len(matt_page.cssselect("a")))
print(len(matt_page.cssselect("table")))

# These prints can get long; consider printing just the first ~600 chars in class.
matt_main_text = (matt_page.cssselect("main")[0].text_content().strip() + " ")[:600]
print(matt_main_text)

matt_header_text = (matt_page.cssselect("header")[0].text_content().strip() + " ")[:600]
print(matt_header_text)

matt_nav_text = (matt_page.cssselect("nav")[0].text_content().strip() + " ")[:600]
print(matt_nav_text)

# 2) Pull the full page text (useful for regex extraction)
matt_text = matt_page.cssselect("body")[0].text_content()

# 3) Extract a job title line (regex)
# This pattern tries to capture a chunk of text containing "Professor ...".
title_pattern = r"(?:Distinguished|Liberal Arts|Roy C\.|Arnold S\.|James P\.)?\s*(?:Associate\s+)?Professor[^\n\r]{0,120}"
matt_title = (re.findall(title_pattern, matt_text) + [None])[0]
matt_title = re.sub(r"\s+", " ", (matt_title or "")).strip() or None
print(matt_title)

# 4) Extract a PSU email address (regex)
email_pattern = r"[A-Za-z0-9._%+-]+@psu\.edu"
matt_email = (re.findall(email_pattern, matt_text) + [None])[0]
print(matt_email)

# 5) Extract "Areas of Interest" (HTML via XPath)
matt_areas_nodes = matt_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
matt_areas = [n.text_content().strip() for n in matt_areas_nodes]
print(matt_areas)

# 6) Extract "Bio" (HTML via XPath union)
# (Some sites use h2, others use h3 — we grab both without deciding which is “right”.)
matt_bio_nodes = matt_page.xpath(
    "//h2[normalize-space()='Professional Bio']/following-sibling::*[1]"
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]"
)
matt_bio = "\n".join([re.sub(r"\s+", " ", n.text_content()).strip() for n in matt_bio_nodes])
print(matt_bio)

# 7) Combine whatever we found into one string (semicolon-separated)
matt_interests = "; ".join(matt_areas)

# 8) Count how many interest items we captured
matt_n_interest_items = len(matt_areas)

# 9) Store results in a single row (dict -> later becomes DataFrame)
matt_row = {
    "name": matt_name,
    "department": matt_dept,
    "url": matt_url,
    "scraped_title": matt_title,
    "scraped_email": matt_email,
    "scraped_interests": matt_interests,
    "n_interest_items": matt_n_interest_items,
    "bio": matt_bio,
}

# -----------------------------------------------------------------------------
# Step 2: Scrape Sona N. Golder (repeat the same workflow)
# -----------------------------------------------------------------------------
sona_resp = requests.get(sona_url, headers=HEADERS)
sona_page = html.fromstring(sona_resp.content)

sona_text = sona_page.cssselect("body")[0].text_content()

sona_title = (re.findall(title_pattern, sona_text) + [None])[0]
sona_title = re.sub(r"\s+", " ", (sona_title or "")).strip() or None

sona_email = (re.findall(email_pattern, sona_text) + [None])[0]

sona_areas_nodes = sona_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
sona_areas = [n.text_content().strip() for n in sona_areas_nodes]

sona_bio_nodes = sona_page.xpath(
    "//h2[normalize-space()='Professional Bio']/following-sibling::*[1]"
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]"
)
sona_bio = "\n".join([re.sub(r"\s+", " ", n.text_content()).strip() for n in sona_bio_nodes])

sona_interests = "; ".join(sona_areas)
sona_n_interest_items = len(sona_areas)

sona_row = {
    "name": sona_name,
    "department": sona_dept,
    "url": sona_url,
    "scraped_title": sona_title,
    "scraped_email": sona_email,
    "scraped_interests": sona_interests,
    "n_interest_items": sona_n_interest_items,
    "bio": sona_bio,
}

# -----------------------------------------------------------------------------
# Step 3: Scrape Derek Kreager (repeat the same workflow)
# -----------------------------------------------------------------------------
derek_resp = requests.get(derek_url, headers=HEADERS)
derek_page = html.fromstring(derek_resp.content)

derek_text = derek_page.cssselect("body")[0].text_content()

derek_title = (re.findall(title_pattern, derek_text) + [None])[0]
derek_title = re.sub(r"\s+", " ", (derek_title or "")).strip() or None

derek_email = (re.findall(email_pattern, derek_text) + [None])[0]

# Note: Derek's page uses "Research Interests" as the list header in your R script.
derek_areas_nodes = derek_page.xpath("//h2[normalize-space()='Research Interests']/following-sibling::ul[1]/li")
derek_areas = [n.text_content().strip() for n in derek_areas_nodes]

derek_bio_nodes = derek_page.xpath(
    "//h2[normalize-space()='Professional Bio']/following-sibling::*[1]"
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]"
)
derek_bio_texts = [re.sub(r"\s+", " ", n.text_content()).strip() for n in derek_bio_nodes]

# Replicates the idea of derek_bio[2] in R (second element if present; otherwise None)
derek_bio = (derek_bio_texts + [None, None])[1]

derek_interests = "; ".join(derek_areas)
derek_n_interest_items = len(derek_areas)

derek_row = {
    "name": derek_name,
    "department": derek_dept,
    "url": derek_url,
    "scraped_title": derek_title,
    "scraped_email": derek_email,
    "scraped_interests": derek_interests,
    "n_interest_items": derek_n_interest_items,
    "bio": derek_bio,
}

# -----------------------------------------------------------------------------
# Step 5: Combine the scraped rows into one data frame and inspect
# -----------------------------------------------------------------------------
scraped_profiles = pd.DataFrame([matt_row, sona_row, derek_row])

print(scraped_profiles)

# -----------------------------------------------------------------------------
# Step 6: Quick plot (interest items captured per faculty member)
# -----------------------------------------------------------------------------
plot_df = scraped_profiles.sort_values("n_interest_items")

plt.figure()
plt.barh(plot_df["name"], plot_df["n_interest_items"])
plt.title("Interest Items Captured from PSU Profile Pages")
plt.xlabel("Number of interest items captured")
plt.ylabel("Faculty member")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Part 2: Pulling Google Scholar Data (Citations Over Time)
# -----------------------------------------------------------------------------
# Goal:
# - For each professor, we will:
#   (1) Define the Google Scholar ID
#   (2) Pull a profile summary
#   (3) Pull publications (and view the first 5)
#   (4) Pull citation history by year
#   (5) Combine all citation histories into one table and plot them

# -----------------------------------------------------------------------------
# Step 1: Hard-code Google Scholar IDs
# -----------------------------------------------------------------------------
matt_scholar_id  = "yPbxmSwAAAAJ"
sona_scholar_id  = "Cuz1fTcAAAAJ"
derek_scholar_id = "9c6_ChYAAAAJ"

# -----------------------------------------------------------------------------
# Step 2: Pull Google Scholar profiles (sequentially)
# -----------------------------------------------------------------------------
matt_profile = scholarly.fill(scholarly.search_author_id(matt_scholar_id))
sona_profile = scholarly.fill(scholarly.search_author_id(sona_scholar_id))
derek_profile = scholarly.fill(scholarly.search_author_id(derek_scholar_id))

print("\n------------------------------")
print("Google Scholar Profile Summaries")
print("------------------------------\n")

print(matt_name)
print(matt_profile)

print("\n" + sona_name)
print(sona_profile)

print("\n" + derek_name)
print(derek_profile)

# -----------------------------------------------------------------------------
# Step 3: Pull Google Scholar publications (sequentially)
# -----------------------------------------------------------------------------
# Note: scholarly returns a list of publications inside each profile dict.
# We'll convert to a DataFrame and print the first 5 rows.

matt_pubs = pd.DataFrame([
    {
        "title": p.get("bib", {}).get("title"),
        "year": p.get("bib", {}).get("pub_year"),
        "citations": p.get("num_citations"),
    }
    for p in matt_profile.get("publications", [])
])

sona_pubs = pd.DataFrame([
    {
        "title": p.get("bib", {}).get("title"),
        "year": p.get("bib", {}).get("pub_year"),
        "citations": p.get("num_citations"),
    }
    for p in sona_profile.get("publications", [])
])

derek_pubs = pd.DataFrame([
    {
        "title": p.get("bib", {}).get("title"),
        "year": p.get("bib", {}).get("pub_year"),
        "citations": p.get("num_citations"),
    }
    for p in derek_profile.get("publications", [])
])

print("\n------------------------------")
print("Recent Publications (first 5)")
print("------------------------------\n")

print(matt_name)
print(matt_pubs.head(5))

print("\n" + sona_name)
print(sona_pubs.head(5))

print("\n" + derek_name)
print(derek_pubs.head(5))

# -----------------------------------------------------------------------------
# Step 4: Pull citation history (citations by year) and combine
# -----------------------------------------------------------------------------
matt_cites_per_year = matt_profile.get("cites_per_year", {})
sona_cites_per_year = sona_profile.get("cites_per_year", {})
derek_cites_per_year = derek_profile.get("cites_per_year", {})

matt_ct = pd.DataFrame({"year": list(matt_cites_per_year.keys()), "cites": list(matt_cites_per_year.values())})
matt_ct["name"] = matt_name

sona_ct = pd.DataFrame({"year": list(sona_cites_per_year.keys()), "cites": list(sona_cites_per_year.values())})
sona_ct["name"] = sona_name

derek_ct = pd.DataFrame({"year": list(derek_cites_per_year.keys()), "cites": list(derek_cites_per_year.values())})
derek_ct["name"] = derek_name

citation_df = pd.concat([matt_ct, sona_ct, derek_ct], ignore_index=True)
citation_df = citation_df.sort_values(["name", "year"])

print(citation_df.head(10))

# -----------------------------------------------------------------------------
# Step 5: Plot citations over time for each professor
# -----------------------------------------------------------------------------
plt.figure()
plt.plot(matt_ct["year"], matt_ct["cites"], marker="o", label=matt_name)
plt.plot(sona_ct["year"], sona_ct["cites"], marker="o", label=sona_name)
plt.plot(derek_ct["year"], derek_ct["cites"], marker="o", label=derek_name)

plt.title("Google Scholar Citation History (Recent Years)")
plt.xlabel("Year")
plt.ylabel("Citations")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Step 6: Median citations per year for each professor
# -----------------------------------------------------------------------------
median_cites = citation_df.groupby("name", as_index=False)["cites"].median()
median_cites = median_cites.rename(columns={"cites": "median_cites"})

print(median_cites)
