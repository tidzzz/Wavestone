# dbscan_miner.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from base_miner import BaseRoleMiner

class DBSCANRoleMiner(BaseRoleMiner):
    """
    DBSCAN-based Role Mining
    
    Parameters:
    -----------
    similarity_threshold : float (0-1)
        Controls how similar users must be to be grouped together
        Internally converted to eps (distance metric)
        Higher value = more similar users required = fewer, tighter roles
        
    grouping_threshold : int
        Minimum number of users to form a valid role (maps to min_samples)
        
    impact_threshold : float (0-1)
        Minimum % of users that must have a permission for it to be in role
    """
    
    def __init__(self, similarity_threshold=0.7, grouping_threshold=3, impact_threshold=0.8):
        super().__init__(similarity_threshold, grouping_threshold, impact_threshold)
        self.model = None
        self.suggested_params = None
    
    def tune_parameters(self, metric='jaccard', plot=True, save_plot=None):
        """
        Auto-suggest parameters based on data characteristics
        
        The similarity_threshold will be converted to eps automatically
        
        Parameters:
        -----------
        metric : str
            Distance metric ('jaccard' recommended for binary data)
        plot : bool
            Show k-distance plot
        save_plot : str
            Path to save plot
        """
        if self.data is None:
            raise ValueError("Load data first with load_data()")
        
        print(f"\n Analyzing data characteristics (metric={metric})...")
        
        X = self.data.values
        
        # Use grouping_threshold as min_samples
        min_samples = self.grouping_threshold
        
        # Calculate distances to k-th neighbor
        if min_samples >= len(X):
            min_samples = max(2, len(X) // 10)
            print(f"     Adjusted min_samples to {min_samples} (dataset too small)")
        
        neighbors = NearestNeighbors(n_neighbors=min_samples, metric=metric)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        k_distances = np.sort(distances[:, -1])
        
        # Convert similarity_threshold to eps
        # similarity_threshold is in [0,1], higher = more strict
        # We need to map this to distance (lower distance = higher similarity)
        
        # For Jaccard: distance = 1 - similarity
        # For similarity_threshold = 0.7, we want users with Jaccard similarity >= 0.7
        # This means Jaccard distance <= 0.3
        
        if metric == 'jaccard':
            # Jaccard distance = 1 - Jaccard similarity
            target_distance = 1 - self.similarity_threshold
        elif metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            target_distance = 1 - self.similarity_threshold
        else:
            # For Euclidean, use percentile approach
            percentile = (1 - self.similarity_threshold) * 100
            target_distance = np.percentile(k_distances, percentile)
        
        # Find actual eps value close to target
        eps_percentiles = [50 ,55 ,60 ,65 ,70 ,75, 80, 85, 90, 95]
        suggestions = []
        
        for p in eps_percentiles:
            eps_candidate = np.percentile(k_distances, p)
            suggestions.append({
                'eps': round(eps_candidate, 4),
                'min_samples': min_samples,
                'percentile': p,
                'reasoning': f'{p}th percentile of {min_samples}-distances'
            })
        
        # Also add target distance
        suggestions.insert(0, {
            'eps': round(target_distance, 4),
            'min_samples': min_samples,
            'percentile': None,
            'reasoning': f'Computed from similarity_threshold={self.similarity_threshold}'
        })
        
        print(f"\n Parameter suggestions:")
        print(f"   Based on your thresholds:")
        print(f"   • Similarity threshold: {self.similarity_threshold} → eps ≈ {target_distance:.4f}")
        print(f"   • Grouping threshold: {self.grouping_threshold} → min_samples = {min_samples}")
        
        print(f"\n   Alternative eps values based on data distribution:")
        for i, sug in enumerate(suggestions[1:], 1):
            print(f"   {i}. eps={sug['eps']} ({sug['reasoning']})")
        
        self.suggested_params = suggestions[0]  # Use computed one by default
        
        if plot:
            self._plot_k_distance(k_distances, min_samples, metric, 
                                 target_distance, save_plot)
        
        return suggestions
    
    def _plot_k_distance(self, k_distances, k, metric, target_eps, save_path=None):
        """Plot k-distance graph"""
        plt.figure(figsize=(12, 7))
        plt.plot(k_distances, linewidth=2, color='steelblue')
        plt.xlabel('Data points (sorted by distance)', fontsize=12)
        plt.ylabel(f'{k}-distance', fontsize=12)
        plt.title(f'K-distance Graph (k={k}, metric={metric})\n'
                 f'Similarity threshold={self.similarity_threshold} → Target eps={target_eps:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Target line
        plt.axhline(y=target_eps, color='red', linestyle='--', linewidth=2,
                   label=f'Target eps={target_eps:.4f}', alpha=0.8)
        
        # Percentile lines
        percentiles = [75, 85, 90, 95]
        colors = ['green', 'orange', 'purple', 'brown']
        for p, c in zip(percentiles, colors):
            val = np.percentile(k_distances, p)
            plt.axhline(y=val, color=c, linestyle=':', alpha=0.6, 
                       label=f'{p}th %ile: {val:.4f}')
        
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    K-distance plot → {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def fit(self, eps=None, min_samples=None, metric='jaccard', auto_tune=True, verbose=True):
        """
        Fit DBSCAN model
        
        Parameters:
        -----------
        eps : float
            Maximum distance (auto-computed from similarity_threshold if None)
        min_samples : int
            Minimum samples per cluster (uses grouping_threshold if None)
        metric : str
            Distance metric
        auto_tune : bool
            Whether to auto-tune parameters first
        verbose : bool
            Print progress
        """
        if self.data is None:
            raise ValueError("Load data first with load_data()")
        
        # Auto-tune if requested or if params not provided
        if auto_tune and (eps is None or min_samples is None):
            if self.suggested_params is None:
                self.tune_parameters(metric=metric, plot=False)
            
            if eps is None:
                eps = self.suggested_params['eps']
            if min_samples is None:
                min_samples = self.suggested_params['min_samples']
        else:
            # Use grouping_threshold as default
            if min_samples is None:
                min_samples = self.grouping_threshold
            
            # Compute eps from similarity if not provided
            if eps is None:
                if metric in ['jaccard', 'cosine']:
                    eps = 1 - self.similarity_threshold
                else:
                    eps = 0.5  # Default fallback
        
        if verbose:
            print(f"\n Fitting DBSCAN...")
            print(f"   Algorithm parameters:")
            print(f"   • eps: {eps} (distance threshold)")
            print(f"   • min_samples: {min_samples} (from grouping_threshold)")
            print(f"   • metric: {metric}")
            print(f"\n   Role mining thresholds:")
            print(f"   • Similarity: {self.similarity_threshold}")
            print(f"   • Grouping: {self.grouping_threshold}")
            print(f"   • Impact: {self.impact_threshold}")
        
        # Fit DBSCAN
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
        self.labels = self.model.fit_predict(self.data.values)
        
        # Extract roles (applies grouping & impact thresholds)
        self.extract_roles()
        
        # Evaluate
        self.evaluate()
        
        if verbose:
            print(f"\n Clustering complete:")
            print(f"   • Valid roles: {len(self.roles)}")
            print(f"   • Noise/rejected: {self.metrics['num_noise']} users")
        
        return self
    
    def handle_noise(self, strategy='assign_nearest', min_similarity=0.3):
        """
        Handle noise/rejected users
        
        Parameters:
        -----------
        strategy : str
            'assign_nearest': Assign to most similar role
            'create_micro_roles': Create individual roles
            'flag_for_review': Just report them
        min_similarity : float
            Minimum similarity to assign to a role (for 'assign_nearest')
        """
        noise_mask = self.labels == -1
        noise_users = self.data.index[noise_mask].tolist()
        
        if not noise_users:
            print(" No noise users to handle")
            return
        
        print(f"\n Handling {len(noise_users)} noise/rejected users...")
        print(f"   Strategy: {strategy}")
        
        if strategy == 'assign_nearest':
            assigned = 0
            for user in noise_users:
                user_perms = set(self.data.loc[user][self.data.loc[user] == 1].index)
                
                if not user_perms:
                    continue
                
                # Find most similar role
                best_role = None
                best_similarity = 0
                
                for role_name, role_info in self.roles.items():
                    role_perms = set(role_info['permissions'])
                    if not role_perms:
                        continue
                    
                    # Jaccard similarity
                    intersection = len(user_perms & role_perms)
                    union = len(user_perms | role_perms)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_role = role_name
                
                if best_role and best_similarity >= min_similarity:
                    self.roles[best_role]['users'].append(user)
                    self.roles[best_role]['size'] += 1
                    assigned += 1
                    print(f"   ✓ {user} → {best_role} (similarity: {best_similarity:.2%})")
            
            print(f"\n   Assigned: {assigned}/{len(noise_users)} users")
            print(f"   Remaining noise: {len(noise_users) - assigned} users")
        
        elif strategy == 'create_micro_roles':
            for i, user in enumerate(noise_users):
                user_perms = self.data.loc[user][self.data.loc[user] == 1].index.tolist()
                
                self.roles[f"MicroRole_{i}"] = {
                    'permissions': user_perms,
                    'users': [user],
                    'size': 1,
                    'permission_count': len(user_perms),
                    'cohesion': 1.0,
                    'coverage': 1.0,
                    'permission_frequencies': {p: 1.0 for p in user_perms}
                }
            
            print(f"   ✓ Created {len(noise_users)} micro-roles")
        
        elif strategy == 'flag_for_review':
            print(f"\n     Users flagged for manual review:")
            for user in noise_users[:10]:
                perms = self.data.loc[user][self.data.loc[user] == 1].index.tolist()
                print(f"   • {user}: {len(perms)} permissions")
            if len(noise_users) > 10:
                print(f"   ... and {len(noise_users)-10} more")


# Example usage
if __name__ == "__main__":
    # Initialize with your thresholds
    miner = DBSCANRoleMiner(
        similarity_threshold=0.7,   
        grouping_threshold=5,        
        impact_threshold=0.8         
    )
    
    # Load data
    miner.load_data('out/user_permission_matrix.csv')
    
    # Auto-tune and visualize
    miner.tune_parameters(plot=True, save_plot='graphs/dbscan/dbscan_tuning.png')
    
    # Fit model
    miner.fit()
    
    # Display results
    miner.print_summary()
    
    # Visualize
    miner.visualize_results(save_path='graphs/dbscan/dbscandbscan_clusters.png')
    
    # Adjust thresholds without re-clustering
    print("\n" + "="*80)
    print("Testing different impact threshold...")
    miner.adjust_thresholds(impact_threshold=0.9)
    miner.print_summary()
    
    # Handle noise
    miner.handle_noise(strategy='assign_nearest', min_similarity=0.3)
    
    # Export
    miner.export_results('results/dbscan')