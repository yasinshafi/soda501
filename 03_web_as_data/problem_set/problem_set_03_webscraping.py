# Importing the packages
import os
os.makedirs("outputs/data", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

from bs4 import BeautifulSoup
import requests
import time
import json
import pandas as pd

import re
import requests
import matplotlib.pyplot as plt

from lxml import html
from scholarly import scholarly

###########################################################
######## FINDING PROFILES OF 10 FACULTY MEMBERS ###########
###########################################################

'''
In this section, I write codes to scrape from the website
of the department of political science to collect information
from ten faculty members.
'''

############## PSU POLI SCI PEOPLE ########################
########################################################### 

# Setting up the page URL, cache file, and cache dictionary
PAGE_URL = 'https://polisci.la.psu.edu/people/'
CACHE_FILE_NAME = 'cacheplscpeople_Scrape.json'
CACHE_DICT = {}
print(f"\nTask 1: printing cache dictionary = {CACHE_DICT}\n")

# Function to load cache from the file
def load_cache():
    try:
        with open(CACHE_FILE_NAME, 'r') as cache_file:
            cache = json.load(cache_file)
    except FileNotFoundError:
        cache = {}
    return cache

# Function to save cache to the file
def save_cache(cache):
    with open(CACHE_FILE_NAME, 'w') as cache_file:
        json.dump(cache, cache_file)

# Function to make a request using cache
def make_url_request_using_cache(url, cache):
    if url in cache:
        print("Using cache")
        return cache[url]
    else:
        print("Fetching")
        time.sleep(1)  # Delay to avoid overloading the server
        response = requests.get(url)
        cache[url] = response.text
        save_cache(cache)
        return cache[url]

'''Professor class representing faculty members'''
class Professor:
    def __init__(self, url, name="No Name", address="No Address", designation="No Designation", professional_bio="No Professional Bio", interests=[]):
        self.name = name
        self.address = address
        self.designation = designation
        self.professional_bio = professional_bio
        self.interests = interests
        self.url = url
        
        # Fetch and parse the page content
        url_text = make_url_request_using_cache(self.url, CACHE_DICT)
        faculty_page = BeautifulSoup(url_text, 'html.parser')
        
        # Extracting Name
        name_tag = faculty_page.find('h1', class_='elementor-heading-title')
        if name_tag:
            self.name = name_tag.string
        print(f"\nFaculty name: {self.name}\n")
        
        # Extracting Designation
        designation_tag = faculty_page.find('div', class_='jet-listing-dynamic-repeater__item')
        if designation_tag:
            designation_div = designation_tag.find('div')
            if designation_div:
                self.designation = designation_div.string
        print(f"\nFaculty designation: {self.designation}\n")
        
        # Extracting Address
        address_tag = faculty_page.find('div', class_='jet-listing-dynamic-field__content')
        if address_tag:
            self.address = address_tag.get_text(separator=" ")
        print(f"\nFaculty address: {self.address}\n")
        
        # Extracting Professional Bio
        bio_container = faculty_page.find_all('div', class_='jet-listing-dynamic-field__content')
        if bio_container:
            for container in bio_container:
                bio_paragraph = container.find('p')
                if bio_paragraph:
                    self.professional_bio = bio_paragraph.get_text()
                    break
        print(f"\nFaculty professional bio: {self.professional_bio}\n")
        
        # Extracting Areas of Interest
        interests_container = faculty_page.find('h2', string='Areas of Interest')
        if interests_container:
            interests_ul = interests_container.find_next_sibling('ul')
            if interests_ul:
                interest_items = interests_ul.find_all('li', class_='jet-listing-dynamic-repeater__item')
                self.interests = [item.find('div').get_text() for item in interest_items if item.find('div')]
            else:
                self.interests = []
        else:
            self.interests = []
        print(f"\nFaculty interests: {self.interests}\n")

    def info(self):
        return f"{self.name} ({self.designation}): {self.address} {self.professional_bio}"

#Operations Begin
# Load cache
CACHE_DICT = load_cache()
print(f"\nTask 2: printing cache after running load_cache() = {CACHE_DICT}\n")

# Make a request to get the HTML content of the page
response = requests.get(PAGE_URL)
faculty_soup = BeautifulSoup(response.text, 'html.parser')

# The actual faculty listing is in a div with specific data attributes
faculty_container = faculty_soup.find('div', {'data-element_type': 'widget', 'data-widget_type': 'jet-listing-grid.default'})

if faculty_container:
    professor_links = faculty_container.find_all('a', class_='jet-listing-dynamic-link__link')
    # Take every other link (since each professor has 2 links - one for name, one for "More about")
    # Or just take unique href values
    unique_links = []
    seen_hrefs = set()
    for link in professor_links:
        href = link.get('href')
        if href and href not in seen_hrefs and '/people/' in href:
            unique_links.append(link)
            seen_hrefs.add(href)
            if len(unique_links) >= 10:
                break
    professor_links = unique_links
else:
    professor_links = []

print(f"Found {len(professor_links)} professor links")
for link in professor_links[:3]:
    print(link.get('href'))

print(f"Found {len(professor_links)} professor links")
print(professor_links)

# Accumulate all the professor URLs
# Use a set to avoid duplicates
professor_urls = set()

# Loop through each link and extract the full URL
for link in professor_links:
    href = link.get('href')
    if href and PAGE_URL in href:
        professor_urls.add(href)

# Convert the set back to a list if needed
professor_urls = list(professor_urls)
print(f"\nTask 4: professor_urls = {professor_urls}\n")

# List to store professor data with URLs and other details
professor_data_list = []

# Take only first 10 professors
professor_urls_subset = professor_urls[:10]

# Iterate over each professor URL
for url in professor_urls_subset:
    professor = Professor(url)
    
    # Print for debugging
    print(f"\nProfessor = {professor.name}\n")
    
    # Append the professor data as a dictionary to the list
    professor_data_list.append({
        "URL": url,
        "name": professor.name,
        "address": professor.address,
        "designation": professor.designation,
        "professional_bio": professor.professional_bio,
        "interests": professor.interests
    })

# Convert the list of dictionaries to a pandas DataFrame
professor_df = pd.DataFrame(professor_data_list)

# Write the DataFrame to a CSV file for easier analysis
output_file = 'outputs/data/professor_data.csv'
professor_df.to_csv(output_file, index=False)

print(f"Data successfully written to {output_file}")
print(f"\nTotal professors scraped: {len(professor_data_list)}")
print(professor_data_list)

###########################################################
################ GOOGLE SCHOLAR PROFILES ##################
###########################################################

'''
In this section, I manually find the google scholar user ids
of the selected faculty members and collect their google
scholar profile information through codes.
'''

# -----------------------------------------------------------------------------
# Part 2: Pulling Google Scholar Data (Citations Over Time)
# -----------------------------------------------------------------------------
# Goal:
# - For each of the 10 professors, we will:
#   (1) Define the Google Scholar ID
#   (2) Pull a profile summary
#   (3) Pull citation history by year
#   (4) Combine all citation histories into one table and plot them
#   (5) Calculate median citations per year for each professor

# -----------------------------------------------------------------------------
# Step 1: Hard-code Google Scholar IDs for 10 professors
# -----------------------------------------------------------------------------
# I manually populate these after scraping the professor names
scholar_ids = {
    "Justin Crofoot": "a0ETp8cAAAAJ",
    "Michael Berkman": "Z5M9Rz0AAAAJ",
    "Xun Cao": "w18ZmkEAAAAJ",
    "Christopher Beem": "C-FDhMsAAAAJ",
    "David Bracken": "",
    "Ray Block": "8LCrZXcAAAAJ",
    "John Christman": "eXQ-0lkAAAAJ",
    "Michael J. Nelson": "SrGrUPsAAAAJ",
    "Lee Ann Banaszak": "i_LM_yAAAAAJ",
    "D. Scott Bennett": "dS_KpRIAAAAJ"
}

# -----------------------------------------------------------------------------
# Step 2: Pull Google Scholar profiles and citation data
# -----------------------------------------------------------------------------
all_citation_data = []

for prof_name, scholar_id in scholar_ids.items():
    # Skip if scholar_id is empty
    if not scholar_id or scholar_id == "":
        print(f"\nSkipping {prof_name} - no Google Scholar ID")
        continue
    
    print(f"\nFetching data for {prof_name}...")
    
    # Get profile
    profile = scholarly.fill(scholarly.search_author_id(scholar_id))
    
    # Extract citation history
    cites_per_year = profile.get("cites_per_year", {})
    
    # Create dataframe for this professor
    prof_ct = pd.DataFrame({
        "year": list(cites_per_year.keys()), 
        "cites": list(cites_per_year.values())
    })
    prof_ct["name"] = prof_name
    
    all_citation_data.append(prof_ct)

# Combine all citation data
citation_df = pd.concat(all_citation_data, ignore_index=True)
citation_df = citation_df.sort_values(["name", "year"])

citation_df.to_csv("outputs/data/citations_over_time.csv", index=False)

print("\nCitation data sample:")
print(citation_df.head(20))

###########################################################
################ START OF THE PROBLEM SET #################
###########################################################

'''
Question 3: Using ten Penn State faculty members from your
department(s) or affiliated with SoDA, create a plot of 
citations over time for each professor.

Answer: Because David Bracken does not have an active
Google Scholar profile, I create plots of 9 selected
faculty members.
'''

# -----------------------------------------------------------------------------
# Step 3: Plot citations over time for each professor
# -----------------------------------------------------------------------------

### Citations over time - Figure 1

'''
This is the first approach. All the faculty members' trends are in one
plot.
'''

plt.figure(figsize=(12, 8))

for prof_name in scholar_ids.keys():
    prof_data = citation_df[citation_df["name"] == prof_name]
    plt.plot(prof_data["year"], prof_data["cites"], marker="o", label=prof_name, linewidth=2)

plt.title("Google Scholar Citations Over Time", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Citations", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figures/citations_over_time.png", dpi=300, bbox_inches='tight')
plt.show()

### Citations over time - Figure 2

'''
This is the second approach. Here, 9 different plots are facet-wrapped.
'''

# Get list of professors who have data
profs_with_data = citation_df["name"].unique()

# Calculate grid dimensions
n_profs = len(profs_with_data)
n_cols = 3  # Number of columns in the grid
n_rows = (n_profs + n_cols - 1) // n_cols  # Ceiling division

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()  # Flatten to make indexing easier

for idx, prof_name in enumerate(profs_with_data):
    prof_data = citation_df[citation_df["name"] == prof_name]
    axes[idx].plot(prof_data["year"], prof_data["cites"], marker="o", linewidth=2)
    axes[idx].set_xlabel("Year", fontsize=9)
    axes[idx].set_ylabel(f"{prof_name}\nCitations", fontsize=9)
    axes[idx].grid(True, alpha=0.3)
    
    # Format x-axis to show only integer years
    axes[idx].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Rotate x-axis labels to prevent overlap
    axes[idx].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for idx in range(len(profs_with_data), len(axes)):
    axes[idx].axis('off')

plt.suptitle("Google Scholar Citations Over Time by Professor", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig("outputs/figures/citations_over_time_2.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------
# Step 4: Shared Research Interest of Professors
# -----------------------------------------------------------------------------

'''
Question 4: Visualize or discuss how the work of these professors
overlaps.
'''
import numpy as np
import seaborn as sns

# Create matrix of shared interests
names = professor_df['name'].tolist()
n = len(names)
overlap_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        interests_i = set(professor_df.iloc[i]['interests']) if professor_df.iloc[i]['interests'] else set()
        interests_j = set(professor_df.iloc[j]['interests']) if professor_df.iloc[j]['interests'] else set()
        overlap_matrix[i, j] = len(interests_i.intersection(interests_j))

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(overlap_matrix, xticklabels=names, yticklabels=names, 
            annot=True, fmt='g', cmap='YlOrRd', cbar_kws={'label': 'Number of Shared Interests'})
plt.title("Research Interest Overlap Between Professors")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/figures/interest_overlap_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

'''
Question 5: What is the median citation count (per year) for each
person in the data?

Answer: I choose to include only observed years. Unobserved years
are before someone actively started publishing. And, once someone
starts getting cited, they typically do not have a missing year,
unless they are new. It is reasonable to include the observed
years only.
'''

# -----------------------------------------------------------------------------
# Step 5: Calculate median citations per year for each professor
# -----------------------------------------------------------------------------
# Note: This computes median over observed years only (years with data in Google Scholar)
median_cites = citation_df.groupby("name", as_index=False)["cites"].median()
median_cites = median_cites.rename(columns={"cites": "median_cites"})

print("\nMedian citations per year (computed over observed years only):")
print(median_cites)

# Save to CSV
median_cites.to_csv("outputs/data/median_citations.csv", index=False)
