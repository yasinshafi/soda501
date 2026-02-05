###############################################################################
# SoDA 501 - Problem Set 3: Text as Data
# Student: Yasin Shafi
# Date: February 5, 2026
#
# This script addresses:
#   Question 5: Word embedding regression (Word2Vec → Ridge regression)
#   Question 6: BERTopic analysis (transformer embeddings → clustering)
###############################################################################

import sys
import os

os.chdir("D:\\r_workspace\\soda_501\\04_text_as_data\\problem_set")

import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import hdbscan

# Reproducibility
np.random.seed(123)

# Create project folders
os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------------------------------
# Set up logging to both console and file
# -----------------------------------------------------------------------------
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        return False
    
    def close(self):
        self.log.close()

# Start logging
logger = Logger("outputs/run_log.txt")
sys.stdout = logger

print("="*80)
print("SoDA 501 - Problem Set 3: Text as Data")
print("Student: Yasin Shafi")
print("Date: February 5, 2026")
print("="*80)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
print("\n--- Loading Data ---")

# Copy data from demo folder if needed
source_data = "../demo/data_raw/week_movie_corpus.csv"
target_data = "data_raw/week_movie_corpus.csv"

if not os.path.exists(target_data) and os.path.exists(source_data):
    print(f"Copying data from {source_data} to {target_data}")
    shutil.copy(source_data, target_data)

# Load the corpus
df = pd.read_csv(target_data)
print(f"Corpus loaded: {df.shape[0]} documents")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# Check outcome variable
print(f"\nOutcome variable distribution:")
print(df["y_outcome"].value_counts())

###############################################################################
# QUESTION 5: Word Embedding Regression
###############################################################################
print("\n" + "="*80)
print("QUESTION 5: Word Embedding Regression (Word2Vec → Ridge Regression)")
print("="*80)

# -----------------------------------------------------------------------------
# Step 1: Tokenization
# -----------------------------------------------------------------------------
print("\n--- Step 1: Tokenizing movie plots ---")

tokenized_docs = []
for text in df["text"].tolist():
    # Simple tokenization: lowercase, extract alphabetic tokens
    tokens = re.findall(r"[a-z]+", text.lower())
    tokenized_docs.append(tokens)

print(f"Number of documents tokenized: {len(tokenized_docs)}")
print(f"Example tokens (first 20): {tokenized_docs[0][:20]}")

# -----------------------------------------------------------------------------
# Step 2: Train Word2Vec model
# -----------------------------------------------------------------------------
print("\n--- Step 2: Training Word2Vec model ---")

w2v = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,  # skip-gram
    seed=123
)

print(f"Word2Vec model trained")
print(f"Vocabulary size: {len(w2v.wv.index_to_key)}")
print(f"Vector dimensionality: {w2v.vector_size}")
print(f"Example words in vocabulary: {w2v.wv.index_to_key[:20]}")

# -----------------------------------------------------------------------------
# Step 3: Construct document-level embeddings
# -----------------------------------------------------------------------------
print("\n--- Step 3: Constructing document-level embeddings ---")

doc_vectors = []
for tokens in tokenized_docs:
    # Filter tokens that are in vocabulary
    tokens_in_vocab = [t for t in tokens if t in w2v.wv]
    
    if len(tokens_in_vocab) == 0:
        # If no tokens in vocab, use zero vector
        doc_vec = np.zeros(w2v.vector_size)
    else:
        # Average word vectors for all tokens in document
        vecs = w2v.wv[tokens_in_vocab]
        doc_vec = vecs.mean(axis=0)
    
    doc_vectors.append(doc_vec)

doc_vectors = np.vstack(doc_vectors)
print(f"Document embedding matrix shape: {doc_vectors.shape}")
print(f"Each document represented as {doc_vectors.shape[1]}-dimensional vector")

# -----------------------------------------------------------------------------
# Step 4: Ridge regression with train/test split
# -----------------------------------------------------------------------------
print("\n--- Step 4: Fitting Ridge regression model ---")

X_train, X_test, y_train, y_test = train_test_split(
    doc_vectors, 
    df["y_outcome"].values, 
    test_size=0.25, 
    random_state=123
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

ridge = Ridge(alpha=1.0, random_state=123)
ridge.fit(X_train, y_train)

print("Ridge regression model fitted")

# -----------------------------------------------------------------------------
# Step 5: Evaluate out-of-sample performance
# -----------------------------------------------------------------------------
print("\n--- Step 5: Out-of-sample performance ---")

y_pred = ridge.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Save metrics
metrics = pd.DataFrame({
    "model": ["word2vec_ridge"],
    "mae": [mae],
    "rmse": [rmse],
    "r2": [r2]
})
metrics.to_csv("outputs/word2vec_regression_metrics.csv", index=False)
print("\nMetrics saved to outputs/word2vec_regression_metrics.csv")

# -----------------------------------------------------------------------------
# Step 6: Create predicted-vs-actual diagnostic plot
# -----------------------------------------------------------------------------
print("\n--- Step 6: Creating diagnostic plot ---")

plt.figure(figsize=(8, 6))
sns.regplot(
    x=y_test, 
    y=y_pred, 
    scatter_kws={'alpha': 0.5}, 
    line_kws={'color': 'red'}
)
plt.xlabel("Actual Outcome (y_test)", fontsize=12)
plt.ylabel("Predicted Outcome (y_pred)", fontsize=12)
plt.title("Ridge Regression: Actual vs. Predicted Values\n(Word2Vec Embeddings)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/word2vec_regression_actual_vs_predicted.png", dpi=200)
print("Diagnostic plot saved to figures/word2vec_regression_actual_vs_predicted.png")
# plt.show()
plt.close()

###############################################################################
# QUESTION 6: BERTopic Analysis
###############################################################################
print("\n" + "="*80)
print("QUESTION 6: BERTopic Analysis (Transformer Embeddings → Clustering)")
print("="*80)

# -----------------------------------------------------------------------------
# Step 1: Generate transformer embeddings
# -----------------------------------------------------------------------------
print("\n--- Step 1: Generating transformer embeddings ---")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(df["text"].tolist(), show_progress_bar=True)

print(f"Embedding matrix shape: {embeddings.shape}")
print(f"Each document represented as {embeddings.shape[1]}-dimensional vector")

# -----------------------------------------------------------------------------
# Step 2: Configure and fit BERTopic model
# -----------------------------------------------------------------------------
print("\n--- Step 2: Fitting BERTopic model ---")

# Configure UMAP for dimensionality reduction
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=123
)

# Configure HDBSCAN for clustering
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# Initialize BERTopic
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True
)

# Fit and transform
topics, probs = topic_model.fit_transform(df["text"].tolist(), embeddings)

print("BERTopic model fitted and transformed")

# -----------------------------------------------------------------------------
# Step 3: Generate topic summary table
# -----------------------------------------------------------------------------
print("\n--- Step 3: Topic summary ---")

topic_info = topic_model.get_topic_info()
print("\nTopic Info:")
print(topic_info)

# Save topic info
topic_info.to_csv("outputs/bertopic_topic_info.csv", index=False)
print("\nTopic info saved to outputs/bertopic_topic_info.csv")

# Add topic assignments to dataframe
df_bert = df.copy()
df_bert["bertopic_topic"] = topics
df_bert["bertopic_max_prob"] = np.max(probs, axis=1)
df_bert.to_csv("data_processed/corpus_with_bertopic.csv", index=False)

# -----------------------------------------------------------------------------
# Step 4: Report key statistics
# -----------------------------------------------------------------------------
print("\n--- Step 4: Key statistics ---")

# Number of topics (excluding outliers)
n_topics_excl_outliers = len(topic_info[topic_info["Topic"] != -1])
print(f"Number of topics discovered (excluding outliers): {n_topics_excl_outliers}")

# Outlier statistics
outlier_count = np.sum(np.array(topics) == -1)

# Outlier statistics
topics_array = np.array(topics)
outlier_count = (topics_array == -1).sum()
outlier_share = outlier_count / len(topics)
print(f"Number of outlier documents (Topic = -1): {outlier_count}")
print(f"Share of outlier documents: {outlier_share:.2%}")

# Topic distribution
print("\nTopic distribution:")
topic_counts = pd.Series(topics).value_counts().sort_index()
print(topic_counts)

# -----------------------------------------------------------------------------
# Step 5: Create topic count plot
# -----------------------------------------------------------------------------
print("\n--- Step 5: Creating topic count plot ---")

# Get topic counts excluding outliers
topic_counts_excl = topic_info[topic_info["Topic"] != -1].copy()

plt.figure(figsize=(10, 5))
plt.bar(
    topic_counts_excl["Topic"].astype(str), 
    topic_counts_excl["Count"],
    color='steelblue',
    alpha=0.8
)
plt.xlabel("Topic", fontsize=12)
plt.ylabel("Number of Documents", fontsize=12)
plt.title("BERTopic: Documents per Topic (Excluding Outliers)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/bertopic_topic_counts.png", dpi=200)
print("Topic count plot saved to figures/bertopic_topic_counts.png")
plt.close()

# Create visualization with all topics including outliers
plt.figure(figsize=(10, 5))
plt.bar(
    topic_info["Topic"].astype(str), 
    topic_info["Count"],
    color=['red' if t == -1 else 'steelblue' for t in topic_info["Topic"]],
    alpha=0.8
)
plt.xlabel("Topic", fontsize=12)
plt.ylabel("Number of Documents", fontsize=12)
plt.title("BERTopic: Documents per Topic (Including Outliers in Red)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/bertopic_topic_counts_with_outliers.png", dpi=200)
print("Topic count plot (with outliers) saved to figures/bertopic_topic_counts_with_outliers.png")
plt.close()

# -----------------------------------------------------------------------------
# Generate topic hierarchy visualization
# -----------------------------------------------------------------------------
print("\n--- Generating additional visualizations ---")

try:
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html("outputs/bertopic_hierarchy.html")
    print("Topic hierarchy saved to outputs/bertopic_hierarchy.html")
except Exception as e:
    print(f"Could not generate hierarchy visualization: {e}")

try:
    fig_docs = topic_model.visualize_documents(
        df["text"].tolist(), 
        embeddings=embeddings
    )
    fig_docs.write_html("outputs/bertopic_document_map.html")
    print("Document map saved to outputs/bertopic_document_map.html")
except Exception as e:
    print(f"Could not generate document map: {e}")

###############################################################################
# IN-CLASS EXTENSION: Multinomial Outcome Classification
###############################################################################
print("\n" + "="*80)
print("IN-CLASS EXTENSION: Multinomial Outcome Classification")
print("="*80)

# -----------------------------------------------------------------------------
# Step 1: Redefine outcome variable as multinomial
# -----------------------------------------------------------------------------
print("\n--- Step 1: Creating multinomial outcome from genres ---")

# Map genre labels to integers
genre_to_label = {"action": 0, "comedy": 1, "drama": 2, "horror": 3, "other": 4}
y_multiclass = df["true_topic"].map(genre_to_label).values

print(f"Genre mapping: {genre_to_label}")
print(f"\nGenre distribution:")
print(pd.Series(y_multiclass).value_counts().sort_index())

# Verify no missing values
if np.any(pd.isna(y_multiclass)):
    print("Warning: Some genres could not be mapped. Removing those rows.")
    valid_idx = ~pd.isna(y_multiclass)
    y_multiclass = y_multiclass[valid_idx]
    doc_vectors_multi = doc_vectors[valid_idx]
else:
    doc_vectors_multi = doc_vectors

# -----------------------------------------------------------------------------
# Step 2: Train/test split
# -----------------------------------------------------------------------------
print("\n--- Step 2: Train/test split for multinomial classification ---")

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    doc_vectors_multi,
    y_multiclass,
    test_size=0.25,
    random_state=123,
    stratify=y_multiclass  # Stratify to maintain class balance
)

print(f"Training set size: {X_train_multi.shape[0]}")
print(f"Test set size: {X_test_multi.shape[0]}")
print(f"\nTraining set class distribution:")
print(pd.Series(y_train_multi).value_counts().sort_index())

# -----------------------------------------------------------------------------
# Step 3: Fit multinomial logistic regression
# -----------------------------------------------------------------------------
print("\n--- Step 3: Fitting multinomial logistic regression ---")

from sklearn.linear_model import LogisticRegression

# Multinomial logistic regression
multi_logreg = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=123
)

multi_logreg.fit(X_train_multi, y_train_multi)
print("Multinomial logistic regression model fitted")

# -----------------------------------------------------------------------------
# Step 4: Predict and evaluate
# -----------------------------------------------------------------------------
print("\n--- Step 4: Prediction and evaluation ---")

y_pred_multi = multi_logreg.predict(X_test_multi)

# Compute accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test_multi, y_pred_multi)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
label_names = ["Action", "Comedy", "Drama", "Horror", "Other"]
print("\nClassification Report:")
print(classification_report(y_test_multi, y_pred_multi, target_names=label_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_test_multi, y_pred_multi)
print("\nConfusion Matrix:")
print(conf_matrix)

# Save results
multi_results = pd.DataFrame({
    "model": ["multinomial_logistic"],
    "accuracy": [accuracy]
})
multi_results.to_csv("outputs/multinomial_classification_metrics.csv", index=False)

# Save confusion matrix
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=label_names,
    columns=label_names
)
conf_matrix_df.to_csv("outputs/confusion_matrix.csv")
print("\nResults saved to outputs/multinomial_classification_metrics.csv")
print("Confusion matrix saved to outputs/confusion_matrix.csv")

# -----------------------------------------------------------------------------
# Step 5: Visualize confusion matrix
# -----------------------------------------------------------------------------
print("\n--- Step 5: Visualizing confusion matrix ---")

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_names,
    yticklabels=label_names,
    cbar_kws={'label': 'Count'}
)
plt.xlabel("Predicted Genre", fontsize=12)
plt.ylabel("True Genre", fontsize=12)
plt.title("Confusion Matrix: Multinomial Genre Classification\n(Word2Vec Embeddings)", fontsize=14)
plt.tight_layout()
plt.savefig("figures/multinomial_confusion_matrix.png", dpi=200)
print("Confusion matrix plot saved to figures/multinomial_confusion_matrix.png")
plt.close()

# Plot per-class accuracy
print("\n--- Per-class performance ---")
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_test_multi, 
    y_pred_multi,
    labels=list(range(5))
)

per_class_perf = pd.DataFrame({
    "Genre": label_names,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "Support": support
})
print(per_class_perf)
per_class_perf.to_csv("outputs/per_class_performance.csv", index=False)

# Visualize per-class F1 scores
plt.figure(figsize=(10, 6))
plt.bar(label_names, f1, color='steelblue', alpha=0.8)
plt.xlabel("Genre", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.title("Per-Class F1-Scores: Multinomial Genre Classification", fontsize=14)
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/per_class_f1_scores.png", dpi=200)
print("Per-class F1 scores plot saved to figures/per_class_f1_scores.png")
plt.close()

print("\n--- Multinomial classification extension complete ---")    

###############################################################################
# COMPLETION
###############################################################################
print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
print("\nOutputs generated:")
print("  - outputs/run_log.txt (this log)")
print("  - outputs/word2vec_regression_metrics.csv")
print("  - outputs/bertopic_topic_info.csv")
print("  - figures/word2vec_regression_actual_vs_predicted.png")
print("  - figures/bertopic_topic_counts.png")
print("  - figures/bertopic_topic_counts_with_outliers.png")
print("  - data_processed/corpus_with_bertopic.csv")
print("\n" + "="*80)

# Close the logger
logger.close()
sys.stdout = sys.__stdout__  # Reset stdout

print("Log file saved. Script execution complete.")


