import torch
import numpy as np

class KMeansCUDA:
    def __init__(self, k=3, max_iters=100, tol=1e-4, device="cuda"):
        # Check if CUDA is available, otherwise fallback to CPU
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def _kmeans_plus_plus(self, X):
        """Intelligent initialization to avoid local optima."""
        n_samples, _ = X.shape
        centroids = torch.empty((self.k, X.shape[1]), device=self.device)
        
        # 1. Randomly pick the first centroid
        idx = torch.randint(0, n_samples, (1,)).item()
        centroids[0] = X[idx]

        for i in range(1, self.k):
            # Compute distances from all points to existing centroids
            dist = torch.cdist(X, centroids[:i], p=2) 
            # Get distance to the nearest centroid for each point
            min_dist, _ = torch.min(dist, dim=1)
            # Pick next centroid with probability proportional to distance squared
            probs = min_dist.pow(2) / min_dist.pow(2).sum()
            custom_dist = torch.distributions.Categorical(probs)
            centroids[i] = X[custom_dist.sample()]
            
        return centroids

    def fit(self, X_numpy):
        # Move data to GPU
        X = torch.tensor(X_numpy, dtype=torch.float32, device=self.device)
        
        # Initialize centroids using K-means++
        self.centroids = self._kmeans_plus_plus(X)

        for i in range(self.max_iters):
            # 1. Assignment Step: Use torch.cdist for high-performance distance calculation
            # cdist is highly optimized for GPU/CUDA architectures
            distances = torch.cdist(X, self.centroids, p=2)
            labels = torch.argmin(distances, dim=1)

            # 2. Update Step: Calculate new means
            new_centroids = torch.stack([
                X[labels == j].mean(0) if (labels == j).any() 
                else self.centroids[j] 
                for j in range(self.k)
            ])

            # 3. Check Convergence
            shift = torch.norm(self.centroids - new_centroids)
            if shift < self.tol:
                print(f"Converged on {self.device} at iteration {i}")
                break
            
            self.centroids = new_centroids

        return labels.cpu().numpy() # Return to CPU for standard processing

# --- Scale Test ---
if __name__ == "__main__":
    # Generate 1 million points to actually feel the CUDA speedup
    data = np.random.randn(1000000, 10).astype(np.float32)
    
    model = KMeansCUDA(k=10)
    labels = model.fit(data)
    print("Final Centroids Shape:", model.centroids.shape)