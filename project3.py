import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

matplotlib.use('Agg')
import warnings

warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix

##
# Output directory for saving figures
##
output_dir = os.path.join(os.path.expanduser('~'), 'Downloads', 'project3_figures')
os.makedirs(output_dir, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches='tight', dpi=150)
    plt.close()

##
# Loading of data
##
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

##
#Pre-processing -- we choose classification
##
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['smoking_status'] = df['smoking_status'].replace({
    'formerly smoked': 2, 'smokes': 1, 'never smoked': 0, 'Unknown': 3
})
df['gender'] = df['gender'].replace({'Male': 2, 'Female': 1, 'Other': 0})
df['ever_married'] = df['ever_married'].replace({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].replace({'Urban': 1, 'Rural': 0})
df = df.drop(columns=['id'])

# Drop rows with missing bmi (clustering cannot operate on NaN; same approach as Project 2).
df_pre = df.dropna(subset=['bmi']).reset_index(drop=True)
print("Shape after dropping bmi NaNs:", df_pre.shape)

# Keep stroke label aside to later use as an EXTERNAL ground truth.
# Assignment explicitly says: do NOT include the label attribute in the clustering inputs.
stroke_labels = df_pre['stroke'].values
X_full = df_pre.drop(columns=['stroke'])

# Project 2 used OHE on 'work_type' while passthrough for the rest.
# For clustering we do the same so distance calculations are not biased by a single multi-level nominal attribute.
nominal_cols = ['work_type']
passthrough_cols = [c for c in X_full.columns if c not in nominal_cols]

preprocessor = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(drop='first', sparse_output=False), nominal_cols),
    ('pass', 'passthrough', passthrough_cols)
])
X_enc = preprocessor.fit_transform(X_full)

ohe_feature_names = list(preprocessor.named_transformers_['ohe'].get_feature_names_out(['work_type']))
feature_names = ohe_feature_names + passthrough_cols
print("Encoded feature names:", feature_names)

# Scale continuous attributes so range doesn't disproportionately influence distance (required by assignment).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enc)
print("Scaled data shape:", X_scaled.shape)

# Some experiements - we should test an unscaled version to compare (assignmet requirement)
####determining which pre processing techniques are useful but not necessary
X_unscaled = X_enc.copy()

##Dimensionality reduction using PCA - for visualization only
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

# Sub-sample for heavy methods (hierarchical on n=4909 → 4909x4909 distance matrix ≈ 180MB; workable but slow).
# We'll use a stratified sample by stroke so both classes are represented.
rng = np.random.RandomState(42)
n_sample = min(1500, len(stroke_labels))
idx_stroke_1 = np.where(stroke_labels == 1)[0]
idx_stroke_0 = np.where(stroke_labels == 0)[0]
n_stroke_1 = min(len(idx_stroke_1), max(1, int(n_sample * len(idx_stroke_1) / len(stroke_labels))))
n_stroke_0 = min(len(idx_stroke_0), n_sample - n_stroke_1)
sample_idx = np.concatenate([
    rng.choice(idx_stroke_1, size=n_stroke_1, replace=False),
    rng.choice(idx_stroke_0, size=n_stroke_0, replace=False),
])
rng.shuffle(sample_idx)
X_sample = X_scaled[sample_idx]
stroke_sample = stroke_labels[sample_idx]
X_pca_sample = X_pca[sample_idx]
print(f"Sample for hierarchical/t-SNE: n={X_sample.shape[0]} (stroke=1: {stroke_sample.sum()})")

##
# Dataset exploration plots
##
# 1a. Scaled feature distribution - confirms pre-processing worked
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot([X_scaled[:, i] for i in range(X_scaled.shape[1])], labels=feature_names)
ax.set_title('Scaled Feature Distributions (post StandardScaler)')
ax.set_ylabel('Scaled Value')
plt.xticks(rotation=45, ha='right')
save_fig('1a_scaled_feature_boxplots')

# 1b. 2D PCA scatter colored by stroke to preview whether structure is visible at all
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=stroke_labels,
                cmap='coolwarm', alpha=0.4, s=10, edgecolor='none')
ax.set_title(f'PCA 2D Projection (colored by stroke label)\n'
             f'Explained variance: {pca.explained_variance_ratio_.sum():.2%}')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(sc, ax=ax, label='Stroke')
save_fig('1b_pca_by_stroke')

##
# SECTION 1: K-MEANS
##
print("=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

# Elbow method - plot SSE for k = 1..10
k_range = range(1, 11)
sse_values = []
sil_values = []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    sse_values.append(km.inertia_)
    if k >= 2:
        sil_values.append(silhouette_score(X_scaled, km.labels_, sample_size=2000, random_state=42))
    else:
        sil_values.append(np.nan)
    print(f"  k={k}: SSE={km.inertia_:.2f}, silhouette={sil_values[-1]:.4f}")

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(list(k_range), sse_values, 'o-', color='steelblue', label='SSE (inertia)')
ax2.plot(list(k_range), sil_values, 's--', color='tomato', label='Silhouette')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('SSE (Inertia)', color='steelblue')
ax2.set_ylabel('Silhouette Coefficient', color='tomato')
ax1.set_title('K-Means: Elbow Plot (SSE) and Silhouette vs k')
ax1.grid(alpha=0.3)
save_fig('2a_kmeans_elbow')

# Pick best k - visual elbow + silhouette peak. We'll use k=3 based on typical elbow behavior, but verify.
best_k_km = int(np.nanargmax(sil_values) + 1)  # +1 because k_range starts at 1 and silhouette[0] is NaN
print(f"Best k by silhouette: {best_k_km}")
# Also keep k=3 for a common elbow interpretation
km_best = KMeans(n_clusters=best_k_km, n_init=10, random_state=42)
km_labels = km_best.fit_predict(X_scaled)

# 2b. Visualize k-means clusters in PCA space
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='tab10',
                alpha=0.5, s=10, edgecolor='none')
axes[0].set_title(f'K-Means clusters (k={best_k_km}) - PCA projection')
axes[0].set_xlabel('PC1');
axes[0].set_ylabel('PC2')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=stroke_labels, cmap='coolwarm',
                alpha=0.5, s=10, edgecolor='none')
axes[1].set_title('Stroke labels - PCA projection')
axes[1].set_xlabel('PC1');
axes[1].set_ylabel('PC2')
save_fig('2b_kmeans_pca_clusters')

# 2c. Silhouette plot for the chosen k
sample_sil = silhouette_samples(X_scaled, km_labels)
fig, ax = plt.subplots(figsize=(8, 5))
y_lower = 10
for i in range(best_k_km):
    cluster_sil = sample_sil[km_labels == i]
    cluster_sil.sort()
    size = cluster_sil.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size, str(i))
    y_lower = y_upper + 10
avg_sil = silhouette_score(X_scaled, km_labels, sample_size=2000, random_state=42)
ax.axvline(avg_sil, color='red', linestyle='--', label=f'mean = {avg_sil:.3f}')
ax.set_title(f'Silhouette plot - K-Means (k={best_k_km})')
ax.set_xlabel('Silhouette coefficient');
ax.set_ylabel('Cluster')
ax.legend()
save_fig('2c_kmeans_silhouette_plot')

##
# SECTION 2: HIERARCHICAL CLUSTERING
##
print("=" * 60)
print("HIERARCHICAL CLUSTERING")
print("=" * 60)
# Run all 4 linkages on the stratified sample
linkages = ['ward', 'complete', 'average', 'single']
hier_results = {}

# 3a. Dendrograms for all 4 linkages
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, link in zip(axes.flat, linkages):
    Z = linkage(X_sample, method=link)
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30, no_labels=True,
               color_threshold=0, above_threshold_color='steelblue')
    ax.set_title(f'Dendrogram - {link} linkage')
    ax.set_xlabel('Sample index (truncated)')
    ax.set_ylabel('Distance')
plt.tight_layout()
save_fig('3a_dendrograms_all_linkages')

# Evaluate each linkage at k = best_k_km for comparison with k-means
for link in linkages:
    hc = AgglomerativeClustering(n_clusters=best_k_km, linkage=link)
    labels = hc.fit_predict(X_sample)
    unique, counts = np.unique(labels, return_counts=True)
    sil = silhouette_score(X_sample, labels) if len(unique) > 1 else np.nan
    hier_results[link] = {'labels': labels, 'silhouette': sil, 'sizes': dict(zip(unique, counts))}
    print(f"  {link:8s} linkage (k={best_k_km}): silhouette={sil:.4f}, cluster sizes={dict(zip(unique, counts))}")

# 3b. Hierarchical clusters shown in PCA space, one subplot per linkage
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, link in zip(axes.flat, linkages):
    labels = hier_results[link]['labels']
    ax.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=labels, cmap='tab10',
               alpha=0.6, s=14, edgecolor='none')
    ax.set_title(f'{link} linkage (k={best_k_km}), silhouette={hier_results[link]["silhouette"]:.3f}')
    ax.set_xlabel('PC1');
    ax.set_ylabel('PC2')
plt.tight_layout()
save_fig('3b_hierarchical_pca_clusters')

# Choose best linkage (highest silhouette, finite)
best_link = max(hier_results.keys(), key=lambda l: (hier_results[l]['silhouette']
                                                    if not np.isnan(hier_results[l]['silhouette']) else -1))
print(f"Best hierarchical linkage: {best_link}")
hier_best_labels = hier_results[best_link]['labels']

##
# SECTION 3: DBSCAN
##
print("=" * 60)
print("DBSCAN CLUSTERING")
print("=" * 60)

# k-distance plot for choosing eps. Heuristic: use k = 2*dim - 1 (recommended).
k_mp = 2 * X_scaled.shape[1]
nbrs = NearestNeighbors(n_neighbors=k_mp).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
kth_dist = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(kth_dist, color='steelblue')
ax.set_title(f'{k_mp}-distance plot (used to select eps)')
ax.set_xlabel('Points sorted by distance to kth NN')
ax.set_ylabel(f'{k_mp}-NN distance')
ax.grid(alpha=0.3)
save_fig('4a_dbscan_kdistance')

# Grid of (eps, min_samples) - search around the elbow of the k-distance plot.
# kth_dist elbow roughly in 2-4 range for scaled stroke data.
dbscan_results = []
for eps in [1.5, 2.0, 2.5, 3.0, 3.5]:
    for min_samples in [5, 10, 20]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        if n_clusters >= 2:
            mask = labels != -1
            sil = silhouette_score(X_scaled[mask], labels[mask], sample_size=2000, random_state=42) \
                if mask.sum() > n_clusters else np.nan
        else:
            sil = np.nan
        dbscan_results.append({
            'eps': eps, 'min_samples': min_samples,
            'n_clusters': n_clusters, 'n_noise': n_noise, 'silhouette': sil
        })
        print(f"  eps={eps}, min_samples={min_samples}: "
              f"clusters={n_clusters}, noise={n_noise}, silhouette={sil}")

dbscan_df = pd.DataFrame(dbscan_results)
print("\nDBSCAN grid search results:")
print(dbscan_df.to_string(index=False))

# Pick best DBSCAN params: most clusters with finite silhouette (tie-break by silhouette).
valid = dbscan_df[(dbscan_df['n_clusters'] >= 2) & (dbscan_df['silhouette'].notna())]
if len(valid) > 0:
    best_row = valid.sort_values(by=['silhouette'], ascending=False).iloc[0]
    best_eps = best_row['eps']
    best_ms = int(best_row['min_samples'])
else:
    # Fallback to middle values if nothing found 2+ clusters
    best_eps, best_ms = 2.0, 10
print(f"Best DBSCAN: eps={best_eps}, min_samples={best_ms}")

db_best = DBSCAN(eps=best_eps, min_samples=best_ms)
db_labels = db_best.fit_predict(X_scaled)
print(f"  → {len(set(db_labels)) - (1 if -1 in db_labels else 0)} clusters, "
      f"{int(np.sum(db_labels == -1))} noise points")

# 4b. Heatmap of DBSCAN grid (silhouette and n_clusters)
pivot_sil = dbscan_df.pivot(index='eps', columns='min_samples', values='silhouette')
pivot_nc = dbscan_df.pivot(index='eps', columns='min_samples', values='n_clusters')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(pivot_sil, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
axes[0].set_title('DBSCAN grid: silhouette')
sns.heatmap(pivot_nc, annot=True, fmt='d', cmap='rocket_r', ax=axes[1])
axes[1].set_title('DBSCAN grid: # clusters')
plt.tight_layout()
save_fig('4b_dbscan_grid_heatmap')

# 4c. DBSCAN clusters in PCA space
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['gray' if l == -1 else plt.cm.tab10(l % 10) for l in db_labels]
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5, s=10, edgecolor='none')
ax.set_title(f'DBSCAN (eps={best_eps}, min_samples={best_ms}) - PCA projection\n'
             f'gray = noise (-1)')
ax.set_xlabel('PC1');
ax.set_ylabel('PC2')
save_fig('4c_dbscan_pca_clusters')

##
# SECTION 4: ANOMALY / OUTLIER DETECTION PER METHOD
##
print("=" * 60)
print("ANOMALY DETECTION")
print("=" * 60)

# 5a. K-Means anomaly: distance from each point to its cluster centroid
centroid_dist = np.linalg.norm(X_scaled - km_best.cluster_centers_[km_labels], axis=1)
km_anom_threshold = np.percentile(centroid_dist, 99)  # top 1%
km_outliers = np.where(centroid_dist > km_anom_threshold)[0]
print(f"K-Means outliers (top 1% by centroid distance): {len(km_outliers)} points")

# 5b. Hierarchical anomaly: distance to cluster medoid on the sample
hier_outliers = np.array([], dtype=int)
hier_scores = np.zeros(X_sample.shape[0])
for c in np.unique(hier_best_labels):
    members = np.where(hier_best_labels == c)[0]
    Xc = X_sample[members]
    medoid_idx_local = np.argmin(pdist(Xc).sum() if False else np.sum(squareform(pdist(Xc)), axis=1))
    medoid = Xc[medoid_idx_local]
    dists = np.linalg.norm(Xc - medoid, axis=1)
    hier_scores[members] = dists
hier_threshold = np.percentile(hier_scores, 99)
hier_outliers = np.where(hier_scores > hier_threshold)[0]
print(f"Hierarchical outliers (top 1% by medoid distance, on sample): {len(hier_outliers)} points")

# 5c. DBSCAN anomaly: already defined - noise points (label == -1)
db_outliers = np.where(db_labels == -1)[0]
print(f"DBSCAN outliers (noise label -1): {len(db_outliers)} points")

# 5d. Plot anomaly scores in PCA space (highlight outliers)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# K-Means panel
sc = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=centroid_dist, cmap='plasma',
                     s=10, alpha=0.6, edgecolor='none')
axes[0].scatter(X_pca[km_outliers, 0], X_pca[km_outliers, 1], facecolor='none',
                edgecolor='red', s=40, linewidths=1.2, label=f'outliers (n={len(km_outliers)})')
axes[0].set_title(f'K-Means anomaly score\n(distance to centroid)')
axes[0].set_xlabel('PC1');
axes[0].set_ylabel('PC2');
axes[0].legend()
plt.colorbar(sc, ax=axes[0])
# Hierarchical panel (sample points only)
sc = axes[1].scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=hier_scores,
                     cmap='plasma', s=14, alpha=0.6, edgecolor='none')
axes[1].scatter(X_pca_sample[hier_outliers, 0], X_pca_sample[hier_outliers, 1],
                facecolor='none', edgecolor='red', s=40, linewidths=1.2,
                label=f'outliers (n={len(hier_outliers)})')
axes[1].set_title(f'Hierarchical anomaly score ({best_link})\n(distance to cluster medoid)')
axes[1].set_xlabel('PC1');
axes[1].set_ylabel('PC2');
axes[1].legend()
plt.colorbar(sc, ax=axes[1])
# DBSCAN panel
is_noise = (db_labels == -1).astype(int)
axes[2].scatter(X_pca[~(is_noise.astype(bool)), 0], X_pca[~(is_noise.astype(bool)), 1],
                c='steelblue', s=10, alpha=0.4, label='core/border')
axes[2].scatter(X_pca[db_outliers, 0], X_pca[db_outliers, 1], facecolor='none',
                edgecolor='red', s=40, linewidths=1.2, label=f'noise (n={len(db_outliers)})')
axes[2].set_title('DBSCAN anomaly\n(label == -1)')
axes[2].set_xlabel('PC1');
axes[2].set_ylabel('PC2');
axes[2].legend()
plt.tight_layout()
save_fig('5a_anomaly_overview')


# 5e. Profile outliers vs normal points - stroke rate, age, avg_glucose, bmi
def outlier_profile(outlier_idx_full, label):
    if len(outlier_idx_full) == 0:
        return None
    normal_mask = np.ones(len(df_pre), dtype=bool)
    normal_mask[outlier_idx_full] = False
    profile = pd.DataFrame({
        'attribute': ['stroke_rate', 'age_mean', 'avg_glucose_mean', 'bmi_mean', 'hypertension_rate'],
        'outliers': [
            stroke_labels[outlier_idx_full].mean(),
            df_pre.iloc[outlier_idx_full]['age'].mean(),
            df_pre.iloc[outlier_idx_full]['avg_glucose_level'].mean(),
            df_pre.iloc[outlier_idx_full]['bmi'].mean(),
            df_pre.iloc[outlier_idx_full]['hypertension'].mean(),
        ],
        'normal': [
            stroke_labels[normal_mask].mean(),
            df_pre[normal_mask]['age'].mean(),
            df_pre[normal_mask]['avg_glucose_level'].mean(),
            df_pre[normal_mask]['bmi'].mean(),
            df_pre[normal_mask]['hypertension'].mean(),
        ],
    })
    profile['method'] = label
    return profile


# Map hier_outliers (sample indices) back to full-dataset indices for a fair comparison
hier_outliers_full = sample_idx[hier_outliers]
profiles = pd.concat([
    outlier_profile(km_outliers, 'K-Means'),
    outlier_profile(hier_outliers_full, f'Hier ({best_link})'),
    outlier_profile(db_outliers, 'DBSCAN'),
], ignore_index=True)
print("\nOutlier vs normal profile by method:")
print(profiles.to_string(index=False))

# 5f. Check overlap between outlier sets (sanity on common anomalies)
km_set = set(km_outliers.tolist())
hier_set = set(hier_outliers_full.tolist())
db_set = set(db_outliers.tolist())
print(f"\nOverlap K-Means ∩ Hierarchical: {len(km_set & hier_set)}")
print(f"Overlap K-Means ∩ DBSCAN:       {len(km_set & db_set)}")
print(f"Overlap Hierarchical ∩ DBSCAN:  {len(hier_set & db_set)}")
print(f"Overlap all three:              {len(km_set & hier_set & db_set)}")

##
# SECTION 5: INTERNAL INDICES (SSE, correlation heatmap, silhouette)
##
print("=" * 60)
print("INTERNAL INDICES")
print("=" * 60)

# 6a. Correlation of distance matrix and incidence matrix (on sample for tractability)
dist_matrix = squareform(pdist(X_sample))


def incidence(labels):
    labels = np.asarray(labels)
    return (labels[:, None] == labels[None, :]).astype(int)


# Build three incidence matrices using the SAME sample indices
km_sample_labels = km_best.predict(X_sample)
hier_sample_labels = hier_best_labels  # already computed on X_sample
db_sample_labels = db_best.fit_predict(X_sample)

I_km = incidence(km_sample_labels)
I_hier = incidence(hier_sample_labels)
I_db = incidence(db_sample_labels)


# Correlation between distance matrix and (1 - incidence) since points in same cluster should be close.
def corr_dist_incidence(D, I):
    # Flatten upper triangle only (exclude diagonal)
    iu = np.triu_indices_from(D, k=1)
    d_flat = D[iu]
    i_flat = I[iu]
    return np.corrcoef(d_flat, 1 - i_flat)[0, 1]


corr_km = corr_dist_incidence(dist_matrix, I_km)
corr_hier = corr_dist_incidence(dist_matrix, I_hier)
corr_db = corr_dist_incidence(dist_matrix, I_db)
print(f"Corr(distance, 1-incidence) K-Means:      {corr_km:.4f}")
print(f"Corr(distance, 1-incidence) Hierarchical: {corr_hier:.4f}")
print(f"Corr(distance, 1-incidence) DBSCAN:       {corr_db:.4f}")


# 6b. Heatmap side-by-side: distance matrix reordered by cluster vs incidence matrix
def reorder_by_labels(M, labels):
    order = np.argsort(labels)
    return M[np.ix_(order, order)], order


order_km = np.argsort(km_sample_labels)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(dist_matrix[np.ix_(order_km, order_km)], cmap='viridis', ax=axes[0], cbar=True)
axes[0].set_title(f'Distance matrix reordered by K-Means cluster\n(block-diagonal pattern = good)')
sns.heatmap(I_km[np.ix_(order_km, order_km)], cmap='gray_r', ax=axes[1], cbar=False)
axes[1].set_title('Incidence matrix (same cluster = 1)')
plt.tight_layout()
save_fig('6a_distance_incidence_kmeans')

# Summary SSE table for all three methods
print("\nSSE summary:")
print(f"  K-Means (k={best_k_km}): SSE = {km_best.inertia_:.2f}")


# For hierarchical and DBSCAN we compute SSE manually = sum of squared distances to cluster centroid
def manual_sse(X, labels):
    sse = 0
    for c in np.unique(labels):
        if c == -1:
            continue
        members = X[labels == c]
        centroid = members.mean(axis=0)
        sse += ((members - centroid) ** 2).sum()
    return sse


hier_sse = manual_sse(X_sample, hier_sample_labels)
db_sse = manual_sse(X_scaled, db_labels)
print(f"  Hierarchical ({best_link}, on sample): SSE = {hier_sse:.2f}")
print(f"  DBSCAN (eps={best_eps}, min_samples={best_ms}, noise excluded): SSE = {db_sse:.2f}")

##
# SECTION 6: RELATIVE INDICES (compare pairs of clusterings)
##
print("=" * 60)
print("RELATIVE INDICES")
print("=" * 60)
# Compare K-Means vs each hierarchical linkage at the same k
print(f"\nPair 1: K-Means (k={best_k_km}) vs Hierarchical Ward (k={best_k_km}) - same sample")
# Recompute k-means on the SAMPLE so both clusterings are on the same points
km_on_sample = KMeans(n_clusters=best_k_km, n_init=10, random_state=42).fit(X_sample)
km_sample_lbl = km_on_sample.labels_

relative_rows = []
for link in linkages:
    hc = AgglomerativeClustering(n_clusters=best_k_km, linkage=link)
    lbl = hc.fit_predict(X_sample)
    relative_rows.append({
        'pair': f'K-Means vs Hier-{link}',
        'ARI': adjusted_rand_score(km_sample_lbl, lbl),
        'NMI': normalized_mutual_info_score(km_sample_lbl, lbl),
        'AMI': adjusted_mutual_info_score(km_sample_lbl, lbl),
    })
# Also compare two K-Means runs with different seeds (should be high if k-means is stable)
km_seed2 = KMeans(n_clusters=best_k_km, n_init=10, random_state=1).fit(X_sample).labels_
relative_rows.append({
    'pair': 'K-Means seed=42 vs seed=1',
    'ARI': adjusted_rand_score(km_sample_lbl, km_seed2),
    'NMI': normalized_mutual_info_score(km_sample_lbl, km_seed2),
    'AMI': adjusted_mutual_info_score(km_sample_lbl, km_seed2),
})
# Ward vs Complete (both hierarchical)
ward_lbl = AgglomerativeClustering(n_clusters=best_k_km, linkage='ward').fit_predict(X_sample)
comp_lbl = AgglomerativeClustering(n_clusters=best_k_km, linkage='complete').fit_predict(X_sample)
relative_rows.append({
    'pair': 'Hier-ward vs Hier-complete',
    'ARI': adjusted_rand_score(ward_lbl, comp_lbl),
    'NMI': normalized_mutual_info_score(ward_lbl, comp_lbl),
    'AMI': adjusted_mutual_info_score(ward_lbl, comp_lbl),
})
relative_df = pd.DataFrame(relative_rows)
print(relative_df.to_string(index=False))

# 7a. Bar chart of relative agreement
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(relative_df))
width = 0.25
ax.bar(x_pos - width, relative_df['ARI'], width, label='ARI', color='steelblue')
ax.bar(x_pos, relative_df['NMI'], width, label='NMI', color='seagreen')
ax.bar(x_pos + width, relative_df['AMI'], width, label='AMI', color='tomato')
ax.set_xticks(x_pos)
ax.set_xticklabels(relative_df['pair'], rotation=30, ha='right')
ax.set_ylabel('Score')
ax.set_title('Relative Indices: pairwise clustering agreement')
ax.legend()
ax.set_ylim(0, 1)
save_fig('7a_relative_indices')

##
# SECTION 7: EXTERNAL INDICES vs stroke (+ discretized age as alternative target)
##
print("=" * 60)
print("EXTERNAL INDICES")
print("=" * 60)
# External ground truth A: stroke (binary) - focus target
# External ground truth B: discretized age (young/mid/senior) - tests whether clusters correlate with a different real attribute
age_bins = pd.cut(df_pre['age'], bins=[-0.1, 30, 55, 200],
                  labels=['young', 'mid', 'senior']).astype(str).values

external_rows = []
clusterings = {
    f'K-Means (k={best_k_km})': km_labels,  # full dataset
    f'Hier-{best_link} (k={best_k_km})': None,  # on sample; handled separately
    f'DBSCAN (eps={best_eps}, ms={best_ms})': db_labels,  # full dataset
}
# Handle k-means and dbscan (both run on the full dataset) against full labels
for name, lbl in clusterings.items():
    if lbl is None:  # skip hierarchical for now
        continue
    external_rows.append({
        'clustering': name, 'target': 'stroke',
        'homogeneity': homogeneity_score(stroke_labels, lbl),
        'completeness': completeness_score(stroke_labels, lbl),
        'v_measure': v_measure_score(stroke_labels, lbl),
        'ARI': adjusted_rand_score(stroke_labels, lbl),
        'NMI': normalized_mutual_info_score(stroke_labels, lbl),
    })
    external_rows.append({
        'clustering': name, 'target': 'age_group',
        'homogeneity': homogeneity_score(age_bins, lbl),
        'completeness': completeness_score(age_bins, lbl),
        'v_measure': v_measure_score(age_bins, lbl),
        'ARI': adjusted_rand_score(age_bins, lbl),
        'NMI': normalized_mutual_info_score(age_bins, lbl),
    })
# Hierarchical is only on the sample - compare to sample's stroke/age labels
age_bins_sample = age_bins[sample_idx]
external_rows.append({
    'clustering': f'Hier-{best_link} (sample)', 'target': 'stroke',
    'homogeneity': homogeneity_score(stroke_sample, hier_best_labels),
    'completeness': completeness_score(stroke_sample, hier_best_labels),
    'v_measure': v_measure_score(stroke_sample, hier_best_labels),
    'ARI': adjusted_rand_score(stroke_sample, hier_best_labels),
    'NMI': normalized_mutual_info_score(stroke_sample, hier_best_labels),
})
external_rows.append({
    'clustering': f'Hier-{best_link} (sample)', 'target': 'age_group',
    'homogeneity': homogeneity_score(age_bins_sample, hier_best_labels),
    'completeness': completeness_score(age_bins_sample, hier_best_labels),
    'v_measure': v_measure_score(age_bins_sample, hier_best_labels),
    'ARI': adjusted_rand_score(age_bins_sample, hier_best_labels),
    'NMI': normalized_mutual_info_score(age_bins_sample, hier_best_labels),
})
external_df = pd.DataFrame(external_rows)
print(external_df.to_string(index=False))

# 8a. Grouped bar chart of external indices
fig, ax = plt.subplots(figsize=(12, 5))
pivot = external_df.pivot_table(index='clustering', columns='target', values='v_measure')
pivot.plot(kind='bar', ax=ax, color=['steelblue', 'seagreen'], edgecolor='black')
ax.set_ylabel('V-measure')
ax.set_title('External V-measure: each clustering vs stroke and vs age_group')
ax.set_xticklabels(pivot.index, rotation=25, ha='right')
ax.legend(title='ground truth')
save_fig('8a_external_vmeasure')

# 8b. Contingency matrices for K-Means vs stroke and vs age
for target_name, target in [('stroke', stroke_labels), ('age_group', age_bins)]:
    cm = contingency_matrix(target, km_labels)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'c{i}' for i in range(cm.shape[1])],
                yticklabels=np.unique(target))
    ax.set_title(f'Contingency: K-Means clusters vs {target_name}')
    ax.set_xlabel('Cluster');
    ax.set_ylabel(target_name)
    save_fig(f'8b_contingency_kmeans_{target_name}')

##
# SECTION 8: Effect of pre-processing on clustering (useful vs necessary)
##
print("=" * 60)
print("PRE-PROCESSING EFFECT: scaled vs unscaled")
print("=" * 60)
# Compare silhouette on scaled vs unscaled for the same k
km_unscaled = KMeans(n_clusters=best_k_km, n_init=10, random_state=42).fit(X_unscaled)
sil_scaled = silhouette_score(X_scaled, km_labels, sample_size=2000, random_state=42)
sil_unscaled = silhouette_score(X_unscaled, km_unscaled.labels_, sample_size=2000, random_state=42)
print(f"K-Means silhouette - scaled:   {sil_scaled:.4f}")
print(f"K-Means silhouette - unscaled: {sil_unscaled:.4f}")

# Also compare external v-measure vs stroke
vm_scaled = v_measure_score(stroke_labels, km_labels)
vm_unscaled = v_measure_score(stroke_labels, km_unscaled.labels_)
print(f"V-measure vs stroke - scaled:   {vm_scaled:.4f}")
print(f"V-measure vs stroke - unscaled: {vm_unscaled:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
labels = ['Silhouette', 'V-measure (vs stroke)']
scaled_vals = [sil_scaled, vm_scaled]
unscaled_vals = [sil_unscaled, vm_unscaled]
x_pos = np.arange(len(labels))
width = 0.35
ax.bar(x_pos - width / 2, scaled_vals, width, label='Scaled', color='steelblue', edgecolor='black')
ax.bar(x_pos + width / 2, unscaled_vals, width, label='Unscaled', color='tomato', edgecolor='black')
ax.set_xticks(x_pos);
ax.set_xticklabels(labels)
ax.set_ylabel('Score')
ax.set_title('Effect of StandardScaler on K-Means quality')
ax.legend()
save_fig('9a_preprocessing_effect')

##
# SECTION 9: t-SNE visualization (on sample for tractability)
##
print("Running t-SNE (this may take a minute)...")
t0 = time.time()
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_sample)
print(f"  t-SNE done in {time.time() - t0:.1f}s")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=km_sample_lbl, cmap='tab10', s=14, alpha=0.7)
axes[0].set_title(f'K-Means (k={best_k_km})')
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=hier_best_labels, cmap='tab10', s=14, alpha=0.7)
axes[1].set_title(f'Hierarchical ({best_link})')
colors = ['gray' if l == -1 else plt.cm.tab10(l % 10) for l in db_sample_labels]
axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=14, alpha=0.7)
axes[2].set_title(f'DBSCAN (noise in gray)')
for a in axes:
    a.set_xlabel('t-SNE 1');
    a.set_ylabel('t-SNE 2')
plt.suptitle('t-SNE projection - clusters from the three methods')
plt.tight_layout()
save_fig('10a_tsne_all_methods')

##
# SECTION 10: Cluster member inspection (profile each cluster by domain attribute means)
##
print("=" * 60)
print("CLUSTER PROFILES")
print("=" * 60)
profile_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'smoking_status']
profile_df = df_pre.copy()
profile_df['kmeans_cluster'] = km_labels

cluster_profile = profile_df.groupby('kmeans_cluster')[profile_cols].mean().round(3)
cluster_profile['stroke_rate'] = profile_df.groupby('kmeans_cluster')['stroke'].mean().round(4)
cluster_profile['size'] = profile_df.groupby('kmeans_cluster').size()
print("\nK-Means cluster profile (means of domain attributes):")
print(cluster_profile)

# 11a. Heatmap of cluster profiles (standardized within column for readability)
fig, ax = plt.subplots(figsize=(9, 4))
prof_raw = cluster_profile[profile_cols + ['stroke_rate']]
prof_z = (prof_raw - prof_raw.mean()) / prof_raw.std().replace(0, 1)
annot = prof_raw.round(2).astype(str).values
sns.heatmap(prof_z.values.astype(float), annot=annot, fmt='', cmap='coolwarm',
            center=0, ax=ax,
            xticklabels=prof_raw.columns, yticklabels=prof_raw.index)
ax.set_title('K-Means cluster profiles (z-score across clusters; annotations are raw means)')
ax.set_ylabel('Cluster')
save_fig('11a_kmeans_cluster_profile')

print(f"\nAll figures saved to: {output_dir}")
print("Done.")