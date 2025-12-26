import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

np.random.seed(42)

n_samples = 600

data = pd.DataFrame({
    "years_experience": np.random.randint(0, 8, n_samples),
    "skill_match_pct": np.random.randint(40, 100, n_samples),
    "projects_count": np.random.randint(0, 8, n_samples),
    "internships": np.random.randint(0, 3, n_samples),
    "aptitude_score": np.random.randint(30, 100, n_samples),
    "communication_score": np.random.randint(30, 100, n_samples),
    "resume_score": np.random.randint(40, 100, n_samples),
    "system_design_score": np.random.randint(20, 100, n_samples)
})

print("Dataset Shape:", data.shape)
data.head()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

inertia = []

for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

plt.plot(range(2, 10), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

data["cluster"] = clusters

score = silhouette_score(scaled_data, clusters)
print("Silhouette Score:", score)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

data["pca1"] = pca_data[:, 0]
data["pca2"] = pca_data[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="pca1",
    y="pca2",
    hue="cluster",
    palette="Set2",
    data=data
)
plt.title("Interview Readiness Segmentation")
plt.show()

cluster_summary = data.groupby("cluster").mean()
cluster_summary

cluster_labels = {
    0: "Highly Interview Ready",
    1: "Skill-Gap Focused",
    2: "Strong Technical, Weak Communication",
    3: "Early Career Candidates"
}

data["cluster_label"] = data["cluster"].map(cluster_labels)

sample_candidate = pd.DataFrame({
    "years_experience": [2],
    "skill_match_pct": [75],
    "projects_count": [3],
    "internships": [1],
    "aptitude_score": [70],
    "communication_score": [60],
    "resume_score": [68],
    "system_design_score": [55]
})

sample_scaled = scaler.transform(sample_candidate)
sample_cluster = kmeans.predict(sample_scaled)

print("Assigned Cluster:", sample_cluster[0])
print("Candidate Category:", cluster_labels[sample_cluster[0]])
