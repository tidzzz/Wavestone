# kmeans_miner.py

import numpy as np
## the following two lines can be removed for windows users (wayland-matplotlib comptability)
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from base_miner import BaseRoleMiner

class KMeansRoleMiner(BaseRoleMiner):
    """
    K-means based Role Mining with elbow method
    
    Parameters:
    -----------
    similarity_threshold : float (0-1)
        Not directly used in K-means algorithm, but affects role extraction
        Higher = stricter role definitions
        
    grouping_threshold : int
        Minimum users per role (roles below this are rejected)
        
    impact_threshold : float (0-1)
        Minimum % of users that must have a permission
        for it to be included in the role definition
    """
    
    def __init__(self, similarity_threshold=0.7, grouping_threshold=3, impact_threshold=0.8):
        super().__init__(similarity_threshold, grouping_threshold, impact_threshold)
        self.model = None
        self.suggested_k = None
    
    def tune_parameters(self, k_range=None, plot=True, save_plot=None):
        """
        Find optimal number of clusters using elbow + silhouette
        
        Note: similarity_threshold doesn't affect K-means clustering itself,
        only the final role extraction phase
        
        Parameters:
        -----------
        k_range : range or list
            Range of k to test (auto if None)
        plot : bool
            Show elbow curve
        save_plot : str
            Path to save plot
        """
        if self.data is None:
            raise ValueError("Load data first with load_data()")
        
        print(f"\n Finding optimal number of clusters...")
        print(f"   (Note: similarity_threshold={self.similarity_threshold} will be applied during role extraction)")
        
        X = self.data.values
        n_samples = len(X)
        
        # Auto-determine range
        if k_range is None:
            max_k = min(20, n_samples // 2, n_samples // self.grouping_threshold)
            k_range = range(2, max_k + 1)
        
        print(f"   Testing k from {min(k_range)} to {max(k_range)}...")
        
        inertias = []
        silhouettes = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
        
        # Find elbow using rate of change
        inertias_norm = np.array(inertias)
        inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min())
        
        if len(inertias_norm) > 2:
            # Second derivative
            second_deriv = np.diff(inertias_norm, 2)
            elbow_idx = np.argmax(second_deriv) + 2
        else:
            elbow_idx = 0
        
        elbow_k = list(k_range)[elbow_idx]
        
        # Best silhouette
        silhouette_idx = np.argmax(silhouettes)
        silhouette_k = list(k_range)[silhouette_idx]
        
        print(f"\n Suggested number of clusters:")
        print(f"   • Elbow method: k={elbow_k}")
        print(f"   • Best silhouette: k={silhouette_k} (score={silhouettes[silhouette_idx]:.3f})")
        
        # Choose recommendation
        if abs(elbow_k - silhouette_k) <= 2:
            self.suggested_k = silhouette_k
            print(f"    Recommended: k={self.suggested_k} (best silhouette)")
        else:
            self.suggested_k = elbow_k
            print(f"    Recommended: k={self.suggested_k} (elbow)")
        
        print(f"\n   Remember: grouping_threshold={self.grouping_threshold} will filter out small roles")
        
        if plot:
            self._plot_elbow(k_range, inertias, silhouettes, elbow_k, silhouette_k, save_plot)
        
        return self.suggested_k
    
    def _plot_elbow(self, k_range, inertias, silhouettes, elbow_k, silhouette_k, save_path=None):
        """Plot elbow curve and silhouette scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inertia plot
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=elbow_k, color='red', linestyle='--', linewidth=2,
                   label=f'Elbow at k={elbow_k}')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (WSS)', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Silhouette plot
        ax2.plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=silhouette_k, color='red', linestyle='--', linewidth=2,
                   label=f'Best at k={silhouette_k}')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'K-means Parameter Tuning\n'
                    f'(Thresholds: similarity={self.similarity_threshold}, '
                    f'grouping={self.grouping_threshold}, impact={self.impact_threshold})',
                    fontsize=12, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    Elbow plot → {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def fit(self, n_clusters=None, auto_tune=True, verbose=True):
        """
        Fit K-means model
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (auto-tuned if None)
        auto_tune : bool
            Auto-find optimal k if n_clusters not provided
        verbose : bool
            Print progress
        """
        if self.data is None:
            raise ValueError("Load data first with load_data()")
        
        # Auto-tune if needed
        if n_clusters is None:
            if auto_tune:
                if self.suggested_k is None:
                    self.tune_parameters(plot=False)
                n_clusters = self.suggested_k
            else:
                # Default heuristic
                n_clusters = max(2, len(self.data) // (self.grouping_threshold * 3))
        
        if verbose:
            print(f"\n Fitting K-means...")
            print(f"   Algorithm parameters:")
            print(f"   • n_clusters: {n_clusters}")
            print(f"\n   Role mining thresholds:")
            print(f"   • Similarity: {self.similarity_threshold} (affects role extraction)")
            print(f"   • Grouping: {self.grouping_threshold} (min role size)")
            print(f"   • Impact: {self.impact_threshold} (permission inclusion)")
        
        # Fit K-means
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.model.fit_predict(self.data.values)
        
        # Extract roles (applies thresholds)
        self.extract_roles()
        
        # Evaluate
        self.evaluate()
        
        if verbose:
            print(f"\n Clustering complete:")
            print(f"   • Initial clusters: {n_clusters}")
            print(f"   • Valid roles: {len(self.roles)}")
            print(f"   • Rejected clusters: {n_clusters - len(self.roles)}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Initialize
    miner = KMeansRoleMiner(
        similarity_threshold=0.7,
        grouping_threshold=5,
        impact_threshold=0.8
    )
    
    # Load data
    miner.load_data('out/user_permission_matrix.csv')
    
    # Auto-tune
    miner.tune_parameters(plot=True, save_plot='graphs/kmeans/kmeans_tuning.png')
    
    # Fit
    miner.fit()
    
    # Results
    miner.print_summary()
    miner.visualize_results(save_path='graphs/kmeans/kmeans_clusters.png')
    
    # Try different thresholds
    print("\n" + "="*80)
    print("Testing stricter thresholds...")
    miner.adjust_thresholds(
        grouping_threshold=10,
        impact_threshold=0.9
    )
    miner.print_summary()
    
    # Export
    miner.export_results('results/kmeans')