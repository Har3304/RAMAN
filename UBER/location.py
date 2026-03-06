from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df['pickup_latitude'], df['pickup_longitude'] = np.radians(df['pickup_latitude']), np.radians(df['pickup_longitude'])

X=df[['pickup_latitude', 'pickup_longitude']]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(pca.explained_variance_ratio_)

for c in range(2, 10):
    model = KMeans(n_clusters=c, random_state=42)
    model.fit(X_pca)
    silhoutte_scores.append((c, silhouette_score(X, model.labels_)))
    print(silhoutte_scores)
sorted(silhoutte_scores, key=lambda x: x[1])
print(silhoutte_scores)


model = KMeans(n_clusters=3, random_state=42)
model.fit(X_pca)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
z = np.c_[xx.ravel(), yy.ravel()]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=model.labels_)
plt.contourf(xx, yy, model.predict(z).reshape(xx.shape), alpha=0.4)
plt.show()
