import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        # 1. Randomly initialize centroids from the data points
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. Assign clusters
            # We use broadcasting to calculate distance from each point to each centroid
            # Shape of X: (n_samples, 1, n_features)
            # Shape of centroids: (1, k, n_features)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # 3. Update centroids
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 
                else self.centroids[j] # Handle empty clusters
                for j in range(self.k)
            ])

            # 4. Check for convergence (if centroids move less than tolerance)
            center_shift = np.linalg.norm(self.centroids - new_centroids)
            if center_shift < self.tol:
                print(f"Converged at iteration {i}")
                break
            
            self.centroids = new_centroids

        return labels

# --- Example Usage ---
if __name__ == "__main__":
    # Create synthetic data: 3 clusters in 2D space
    X = np.concatenate([
        np.random.randn(100, 2) + [5, 5],
        np.random.randn(100, 2) + [-5, -5],
        np.random.randn(100, 2) + [5, -5]
    ])

    model = KMeans(k=3)
    labels = model.fit(X)
    print("Final Centroids:\n", model.centroids)