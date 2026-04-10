###############################################################################
# SoDA 501 — Network Data & Graph Workflows
# Problem Set: Q4 (Graph construction + centrality),
#              Q5 (Community detection + evaluation),
#              Q6 (Node classification: LR baseline vs. GCN)
###############################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
np.random.seed(123)
torch.manual_seed(123)

# Output directories
os.makedirs("outputs/figure", exist_ok=True)
os.makedirs("outputs/table", exist_ok=True)

# =============================================================================
# Part 1: Generate Synthetic SBM Network
# =============================================================================
block_sizes = [400, 350, 250]
num_blocks  = len(block_sizes)
num_nodes   = sum(block_sizes)

P = np.array([
    [0.06,  0.01,  0.005],
    [0.01,  0.05,  0.008],
    [0.005, 0.008, 0.04 ]
])

G = nx.stochastic_block_model(block_sizes, P, seed=123)

# Ground-truth community labels
true_labels = np.concatenate([
    np.zeros(block_sizes[0], dtype=int),
    np.ones(block_sizes[1],  dtype=int),
    2 * np.ones(block_sizes[2], dtype=int)
])

# Node features (8-dim, community-specific centers + noise)
num_features = 8
centers = np.array([
    [2, 0, 0, 1, 0, 0, 1, 0],
    [0, 2, 0, 0, 1, 0, 0, 1],
    [0, 0, 2, 0, 0, 1, 1, 0],
], dtype=float)
noise_sd = 0.8

X = np.zeros((num_nodes, num_features), dtype=float)
for k in range(num_blocks):
    mask = true_labels == k
    X[mask] = centers[k] + np.random.normal(0, noise_sd, size=(mask.sum(), num_features))

# =============================================================================
# Q4: Graph Construction + Centrality
# =============================================================================
print("=" * 60)
print("Q4: GRAPH CONSTRUCTION + CENTRALITY")
print("=" * 60)

# --- Basic summary ---
num_edges = G.number_of_edges()
density   = nx.density(G)

print(f"\nGraph summary:")
print(f"  Nodes  : {num_nodes}")
print(f"  Edges  : {num_edges}")
print(f"  Density: {density:.6f}")

summary_df = pd.DataFrame({
    "Metric": ["Nodes", "Edges", "Density"],
    "Value":  [num_nodes, num_edges, f"{density:.6f}"]
})
summary_df.to_csv("outputs/table/q4_graph_summary.csv", index=False)

# --- Edge list round-trip ---
edge_list = pd.DataFrame(list(G.edges()), columns=["u", "v"])
G2 = nx.from_pandas_edgelist(edge_list, source="u", target="v",
                              create_using=nx.Graph())

print(f"\nReconstructed graph — nodes: {G2.number_of_nodes()}, "
      f"edges: {G2.number_of_edges()}")

# --- Centrality measures ---

# Degree
deg_dict      = dict(G2.degree())
deg_series    = pd.Series(deg_dict, name="degree")

# Approximate betweenness (k=200 sample nodes)
betw_dict     = nx.betweenness_centrality(G2, k=200, seed=123)
betw_series   = pd.Series(betw_dict, name="betweenness_approx")

# Eigenvector centrality
eig_dict      = nx.eigenvector_centrality_numpy(G2)
eig_series    = pd.Series(eig_dict, name="eigenvector")

centrality_df = pd.concat([deg_series, betw_series, eig_series], axis=1)
centrality_df.index.name = "node"

# Top-10 per measure
top10_deg  = centrality_df["degree"].nlargest(10)
top10_betw = centrality_df["betweenness_approx"].nlargest(10)
top10_eig  = centrality_df["eigenvector"].nlargest(10)

top10_df = pd.DataFrame({
    "Rank"         : range(1, 11),
    "Degree_node"  : top10_deg.index.tolist(),
    "Degree_val"   : top10_deg.values,
    "Betw_node"    : top10_betw.index.tolist(),
    "Betw_val"     : top10_betw.values,
    "Eig_node"     : top10_eig.index.tolist(),
    "Eig_val"      : top10_eig.values,
})

print("\nTop 10 nodes by each centrality measure:")
print(top10_df.to_string(index=False))
top10_df.to_csv("outputs/table/q4_top10_centrality.csv", index=False)

# --- Figure: three centrality distributions ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].hist(centrality_df["degree"], bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
axes[0].set_title("Degree distribution", fontsize=11)
axes[0].set_xlabel("Degree")
axes[0].set_ylabel("Count of nodes")

axes[1].hist(centrality_df["betweenness_approx"], bins=40, color="darkorange", edgecolor="white", linewidth=0.3)
axes[1].set_title("Approx. betweenness centrality", fontsize=11)
axes[1].set_xlabel("Betweenness (approx, k=200)")
axes[1].set_ylabel("Count of nodes")

axes[2].hist(centrality_df["eigenvector"], bins=40, color="seagreen", edgecolor="white", linewidth=0.3)
axes[2].set_title("Eigenvector centrality", fontsize=11)
axes[2].set_xlabel("Eigenvector centrality")
axes[2].set_ylabel("Count of nodes")

plt.suptitle("Centrality distributions — synthetic SBM (n=1,000)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("outputs/figure/q4_centrality_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: outputs/figure/q4_centrality_distributions.png")

# =============================================================================
# Q5: Community Detection + Evaluation
# =============================================================================
print("\n" + "=" * 60)
print("Q5: COMMUNITY DETECTION + EVALUATION")
print("=" * 60)

louvain_comms  = nx.algorithms.community.louvain_communities(G2, seed=123)
num_louvain    = len(louvain_comms)
louvain_labels = np.zeros(num_nodes, dtype=int)

for comm_id, comm in enumerate(louvain_comms):
    for node in comm:
        louvain_labels[node] = comm_id

louvain_sizes = sorted([len(c) for c in louvain_comms], reverse=True)

print(f"\nNumber of Louvain communities: {num_louvain}")
comm_size_df = pd.DataFrame({
    "Community": [f"C{i}" for i in range(num_louvain)],
    "Size"      : louvain_sizes
})
print("\nCommunity sizes:")
print(comm_size_df.to_string(index=False))
comm_size_df.to_csv("outputs/table/q5_community_sizes.csv", index=False)

ari = adjusted_rand_score(true_labels, louvain_labels)
print(f"\nAdjusted Rand Index (Louvain vs. SBM ground truth): {ari:.4f}")

# --- Figure: community sizes bar chart ---
fig, ax = plt.subplots(figsize=(7, 4))
colors = plt.cm.tab10(np.linspace(0, 1, num_louvain))
ax.bar([f"C{i}" for i in range(num_louvain)], louvain_sizes, color=colors, edgecolor="white")
ax.axhline(block_sizes[0], color="steelblue",  linestyle="--", linewidth=1.2, label=f"True block 0 (n={block_sizes[0]})")
ax.axhline(block_sizes[1], color="darkorange", linestyle="--", linewidth=1.2, label=f"True block 1 (n={block_sizes[1]})")
ax.axhline(block_sizes[2], color="seagreen",   linestyle="--", linewidth=1.2, label=f"True block 2 (n={block_sizes[2]})")
ax.set_title("Louvain community sizes vs. true SBM block sizes", fontsize=11)
ax.set_xlabel("Detected community")
ax.set_ylabel("Number of nodes")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/figure/q5_community_sizes.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure saved: outputs/figure/q5_community_sizes.png")

# =============================================================================
# Q6: Node Classification — LR Baseline vs. GCN
# =============================================================================
print("\n" + "=" * 60)
print("Q6: NODE CLASSIFICATION")
print("=" * 60)

# --- Train/val/test split (60/20/20) ---
perm       = np.random.permutation(num_nodes)
train_size = int(0.60 * num_nodes)
val_size   = int(0.20 * num_nodes)

train_idx = perm[:train_size]
val_idx   = perm[train_size:train_size + val_size]
test_idx  = perm[train_size + val_size:]

y = true_labels.copy()

# --- Logistic regression baseline ---
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X[train_idx], y[train_idx])

val_acc_lr  = accuracy_score(y[val_idx],  lr_model.predict(X[val_idx]))
test_acc_lr = accuracy_score(y[test_idx], lr_model.predict(X[test_idx]))

print(f"\nBaseline (features-only) logistic regression:")
print(f"  Validation accuracy : {val_acc_lr:.4f}")
print(f"  Test accuracy       : {test_acc_lr:.4f}")

# --- GCN setup ---
A_sp    = nx.to_scipy_sparse_array(G2, format="csr", dtype=np.float32)
I_sp    = sp.eye(num_nodes, format="csr", dtype=np.float32)
A_tilde = A_sp + I_sp

deg_tilde    = np.array(A_tilde.sum(axis=1)).flatten()
deg_inv_sqrt = 1.0 / np.sqrt(deg_tilde)
D_inv_sqrt   = sp.diags(deg_inv_sqrt.astype(np.float32), format="csr")
A_norm       = D_inv_sqrt @ A_tilde @ D_inv_sqrt

# Convert to torch sparse
A_coo      = A_norm.tocoo()
A_indices  = torch.tensor(np.vstack((A_coo.row, A_coo.col)), dtype=torch.long)
A_values   = torch.tensor(A_coo.data, dtype=torch.float32)
A_norm_t   = torch.sparse_coo_tensor(A_indices, A_values,
                                      size=(num_nodes, num_nodes)).coalesce()

X_t          = torch.tensor(X, dtype=torch.float32)
y_t          = torch.tensor(y, dtype=torch.long)
train_idx_t  = torch.tensor(train_idx, dtype=torch.long)
val_idx_t    = torch.tensor(val_idx,   dtype=torch.long)
test_idx_t   = torch.tensor(test_idx,  dtype=torch.long)

# GCN parameters
hidden_dim  = 16
num_classes = num_blocks

lin1 = nn.Linear(num_features, hidden_dim)
lin2 = nn.Linear(hidden_dim, num_classes)
optimizer = torch.optim.Adam(
    list(lin1.parameters()) + list(lin2.parameters()),
    lr=0.01, weight_decay=5e-4
)

# Training loop
epochs = 30
history = []

for epoch in range(1, epochs + 1):
    H0     = lin1(X_t)
    H1     = torch.relu(torch.sparse.mm(A_norm_t, H0))
    Z0     = lin2(H1)
    logits = torch.sparse.mm(A_norm_t, Z0)

    loss   = F.cross_entropy(logits[train_idx_t], y_t[train_idx_t])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds     = torch.argmax(logits, dim=1)
    train_acc = (preds[train_idx_t] == y_t[train_idx_t]).float().mean().item()
    val_acc   = (preds[val_idx_t]   == y_t[val_idx_t]).float().mean().item()
    test_acc  = (preds[test_idx_t]  == y_t[test_idx_t]).float().mean().item()

    history.append({
        "epoch"     : epoch,
        "loss"      : float(loss.detach()),
        "train_acc" : train_acc,
        "val_acc"   : val_acc,
        "test_acc"  : test_acc,
    })

    print(f"Epoch {epoch:02d} | loss: {loss.item():.4f} | "
          f"train: {train_acc:.4f} | val: {val_acc:.4f} | test: {test_acc:.4f}")

history_df = pd.DataFrame(history)
history_df.to_csv("outputs/table/q6_training_history.csv", index=False)

val_acc_gcn  = history_df["val_acc"].iloc[-1]
test_acc_gcn = history_df["test_acc"].iloc[-1]

print(f"\nGCN (final epoch):")
print(f"  Validation accuracy : {val_acc_gcn:.4f}")
print(f"  Test accuracy       : {test_acc_gcn:.4f}")

# --- Confusion table ---
preds_np      = preds.detach().cpu().numpy()
test_truth    = y[test_idx]
test_pred_gcn = preds_np[test_idx]

confusion_df  = pd.crosstab(
    pd.Series(test_truth,    name="True label"),
    pd.Series(test_pred_gcn, name="Predicted label")
)
print("\nGCN confusion table (test set):")
print(confusion_df)
confusion_df.to_csv("outputs/table/q6_confusion_table.csv")

# --- Model comparison summary ---
comparison_df = pd.DataFrame({
    "Model"             : ["Logistic Regression (features only)", "GCN (features + graph)"],
    "Validation Acc."   : [f"{val_acc_lr:.4f}",  f"{val_acc_gcn:.4f}"],
    "Test Acc."         : [f"{test_acc_lr:.4f}", f"{test_acc_gcn:.4f}"],
})
print("\nModel comparison:")
print(comparison_df.to_string(index=False))
comparison_df.to_csv("outputs/table/q6_model_comparison.csv", index=False)

# --- Written answer ---
written_q6 = """
Q6 Written Answer (6-10 sentences):

The GCN can outperform the logistic regression baseline in this setting
because it aggregates information from each node's neighbors during the
message-passing steps, effectively using the graph structure as an additional
source of signal beyond the raw node features. In the SBM, within-block
nodes share similar features and are more likely to be connected, so a
node's neighborhood composition is itself informative about community
membership; the GCN exploits that correlation in a way that logistic
regression, which sees only the node's own 8-dimensional feature vector,
cannot. The two-layer design means each node's final representation
integrates information from nodes up to two hops away, which in a
moderately dense SBM covers a substantial portion of the local community.

The baseline can be competitive, or even better, in settings where node
features are highly discriminative relative to the graph signal -- for
example, if the SBM off-diagonal probabilities were higher (noisier
community structure), the graph topology would add more noise than signal,
and message passing would blur rather than sharpen community boundaries.
In those cases, logistic regression's independence from the graph would
actually be an advantage.

A key caution about interpreting high accuracy in social network prediction
tasks is that the network itself was generated from the same ground-truth
labels used for evaluation, so the model is essentially rediscovering
structure that was designed to be recoverable. In applied settings, the
network may have been constructed under representation choices (tie
definitions, thresholds, directionality) that do not correspond to the
theoretical construct of interest, meaning high accuracy on a held-out
test set does not validate the underlying sociological claim -- it only
confirms that the model can predict the constructed label from the
constructed graph, which may be circular.
""".strip()

print("\n" + "=" * 60)
print("All outputs saved to outputs/figure/ and outputs/table/")
print("=" * 60)
