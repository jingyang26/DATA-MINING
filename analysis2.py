import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the dataset
url = 'https://p16-bot-sign-sg.ciciai.com/tos-alisg-i-b2l6bve69y-sg/0a542996a2e74931b9a17cc3fb593feb.csv~tplv-b2l6bve69y-image.image?rk3s=68e6b6b5&x-expires=1719816096&x-signature=hvoff8lKbCLE4lp7Nfk5Cir9yn4%3D'
data = pd.read_csv(url)

# Separate features and target variable
X = data.drop('Diabetes', axis=1)
y = data['Diabetes']

# Feature Importance using RandomForest
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print('Feature Importance:\n', feature_importance)

# Mutual Information
mi = mutual_info_classif(X, y)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print('Mutual Information:\n', mi_series)

# Class Distribution
plt.figure(figsize=(6, 6))
sns.countplot(x='Diabetes', data=data)
plt.title('Class Distribution')
plt.xlabel('Diabetes (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()

# Pairwise Feature Interaction
sns.pairplot(data, hue='Diabetes')
plt.show()

# Correlation with Target Variable
correlation_with_target = data.corr()['Diabetes'].sort_values(ascending=False)
print('Correlation with Target Variable:\n', correlation_with_target)

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA - First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Diabetes')
plt.show()

# t-SNE for Visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Diabetes')
plt.show()