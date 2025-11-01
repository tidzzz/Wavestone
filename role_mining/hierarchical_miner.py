# hierarchical_miner.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from base_miner import BaseRoleMiner

class HierarchicalRoleMiner(BaseRoleMiner):
    """
    Hierarchical clustering for Role Mining
    
    Advantages:
    - Shows role hierarchy (parent-child relationships)
    - No need to specify number of clusters upfront (can cut dendrogram)
    - Good for understanding role structure
    
    Parameters:
    -----------
    similarity_threshold : float (0-1)
        Controls where to cut the dendrogram
        Higher = cut lower = fewer, larger roles
        
    grouping_threshold : int
        Minimum users per role
        
    impact_threshold : float (0-1)
        Permission inclusion threshold
    """
    
    def __init__(self, similarity_threshold=0.7, grouping_threshold=3, impact_threshold=0.8):
        super().__init__(similarity_threshold, grouping_threshold, impact_threshold)
        self.model = None
        self.linkage_matrix = None
        self.suggested_n_clusters = None
    
    def tune_parameters(self, linkage_method='ward', plot=True, save_plot=None):
        """
        Analyze dendrogram to suggest optimal number of clusters
        
        The similarity_threshold is used to determine where to cut the dendrogram
        
        Parameters:
        -----------
        linkage_method : str
            'ward', 'complete', 'average', 'single'
        plot : bool
            Show dendrogram
        save_plot : str
            Path to save dendrogram
        """
        if self.data is None:
            raise ValueError("Load data first with load_data()")
        
        print(f"\n Building dendrogram (linkage={linkage_method})...")
        print(f"   Similarity threshold={self.similarity_threshold} will determine cut height")
        
        X = self.data.values
        
        # Compute linkage matrix
        self.linkage_matrix = linkage(X, method=linkage_method)
        
        # Convert similarity to distance for cutting
        # Higher similarity = lower distance = cut lower in tree
        distance_threshold = 1 - self.similarity_threshold
        
        # Find number of clusters at this distance
        from scipy.cluster.hierarchy import fcluster
        labels_at_threshold = fcluster(
            self.linkage_matrix, 
            t=distance_threshold, 
            criterion='distance'
        )
        n_clusters_at_threshold = len(set(labels_at_threshold))
        
        print(f"\n At similarity_threshold={self.similarity_threshold}:")
        print(f"   • Distance threshold: {distance_threshold:.3f}")
        print(f"   • Suggested clusters: {n_clusters_at_threshold}")
        
        # Also suggest based on inconsistency
        max_clusters = min(20, len(X) // 2)
        inconsistencies = []
        
        for n in range(2, max_clusters + 1):
            labels = fcluster(self.linkage_matrix, t=n, criterion='maxclust')
            # Simple inconsistency measure
            incons = self._compute_inconsistency(labels)
            inconsistencies.append((n, incons))
        
        # Find best by inconsistency (lowest)
        best_by_incons = min(inconsistencies, key=lambda x: x[1])
        
        print(f"   • Best by inconsistency: k={best_by_incons[0]} (score={best_by_incons[1]:.3f})")
        
        self.suggested_n_clusters = n_clusters_at_threshold
        print(f"    Recommended: k={self.suggested_n_clusters}")
        
        if plot:
            self._plot_dendrogram(distance_threshold, save_plot)
        
        return self.suggested_n_clusters
    
    def _compute_inconsistency(self, labels):
        """Simple inconsistency measure"""
        X = self.data.values
        incons = 0
        
        for label in set(labels):
            mask = labels == label
            cluster_data = X[mask]
            
            if len(cluster_data) > 1:
                # Variance within cluster
                incons += np.var(cluster_data)
        
        return incons
    
    def _plot_dendrogram(self, distance_threshold, save_path=None):
        """Plot dendrogram with cut line"""
        plt.figure(figsize=(14, 8))
        
        # Plot dendrogram
        dend = dendrogram(
            self.linkage_matrix,
            truncate_mode='lastp',
            p=30,  # Show last 30 merges
            leaf_font_size=10,
            show_contracted=True
        )
        
        # Add cut line
        plt.axhline(y=distance_threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Cut at distance={distance_threshold:.3f}\n'
                                      f'(similarity={self.similarity_threshold})')
        
        plt.xlabel('Users (or merged clusters)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.title(f'Hierarchical Clustering Dendrogram\n'
                 f'Similarity={self.similarity_threshold}, Grouping≥{self.grouping_threshold}, '
                 f'Impact≥{self.impact_threshold}',
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    Dendrogram → {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def fit(self, n_clusters=None, linkage_method='ward', 
            distance_threshold=None, auto_tune=True, verbose=True):
        """
        Fit Hierarchical clustering
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (auto-tuned if None)
        linkage_method : str
            'ward', 'complete', 'average', 'single'
        distance_threshold : float
            Cut dendrogram at this distance (overrides n_clusters)
        auto_tune : bool
            Auto-find optimal parameters
        verbose : bool
            Print progress
        """
        if self.data is None:
            raise ValueError("Load data first with load_data()")
        
        # Auto-tune if needed
        if n_clusters is None and distance_threshold is None:
            if auto_tune:
                if self.linkage_matrix is None:
                    self.tune_parameters(linkage_method=linkage_method, plot=False)
                n_clusters = self.suggested_n_clusters
            else:
                # Convert similarity to distance
                distance_threshold = 1 - self.similarity_threshold
        
        if verbose:
            print(f"\n Fitting Hierarchical clustering...")
            print(f"   Algorithm parameters:")
            if distance_threshold is not None:
                print(f"   • distance_threshold: {distance_threshold}")
                print(f"     (from similarity_threshold={self.similarity_threshold})")
            else:
                print(f"   • n_clusters: {n_clusters}")
            print(f"   • linkage: {linkage_method}")
            print(f"\n   Role mining thresholds:")
            print(f"   • Grouping: {self.grouping_threshold} (min role size)")
            print(f"   • Impact: {self.impact_threshold} (permission inclusion)")
        
        # Fit model
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage=linkage_method
        )
        
        self.labels = self.model.fit_predict(self.data.values)
        
        # Extract roles
        self.extract_roles()
        
        # Evaluate
        self.evaluate()
        
        if verbose:
            print(f"\n Clustering complete:")
            print(f"   • Clusters formed: {len(set(self.labels))}")
            print(f"   • Valid roles: {len(self.roles)}")
        
        return self
    
    def visualize_hierarchy(self, max_depth=3, save_path=None):
        """
        Visualize role hierarchy as a tree
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth to show
        save_path : str
            Path to save visualization
        """
        if self.linkage_matrix is None:
            print("  No linkage matrix. Run tune_parameters() first.")
            return
        
        print(f"\n Visualizing role hierarchy (depth≤{max_depth})...")
        
        # Create simplified dendrogram
        plt.figure(figsize=(16, 10))
        
        dendrogram(
            self.linkage_matrix,
            truncate_mode='level',
            p=max_depth,
            leaf_font_size=12,
            show_contracted=True
        )
        
        plt.xlabel('Role Groups', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.title(f'Role Hierarchy (max depth={max_depth})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    Hierarchy → {save_path}")
        else:
            plt.show()
        
        plt.close()


# Example usage
if __name__ == "__main__":
    # Initialize
    miner = HierarchicalRoleMiner(
        similarity_threshold=0.7,
        grouping_threshold=5,
        impact_threshold=0.8
    )
    
    # Load data
    miner.load_data('out/user_permission_matrix.csv')
    
    # Auto-tune
    miner.tune_parameters(
        linkage_method='ward',
        plot=True,
        save_plot='graphs/hier/hierarchical_dendrogram.png'
    )
    
    # Fit
    miner.fit()
    
    # Results
    miner.print_summary()
    miner.visualize_results(save_path='graphs/hier/hierarchical_clusters.png')
    
    # Show hierarchy
    miner.visualize_hierarchy(max_depth=4, save_path='graphs/hier/role_hierarchy.png')
    
    # Adjust thresholds
    print("\n" + "="*80)
    print("Testing different similarity threshold...")
    miner.adjust_thresholds(similarity_threshold=0.8)
    miner.print_summary()
    
    # Export
    miner.export_results('results/hierarchical')