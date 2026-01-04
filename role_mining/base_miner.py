# base_miner.py

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
## the following two lines can be removed for windows users (wayland-matplotlib comptability)
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class BaseRoleMiner(ABC):
    """
    Base class for all role mining algorithms
    
    Key Parameters:
    ---------------
    similarity_threshold : float (0-1)
        Minimum similarity between users to be grouped in same role
        Example: 0.7 means users must share 70% of permissions
    
    grouping_threshold : int
        Minimum number of users required to form a role
        Example: 3 means at least 3 users needed per role
    
    impact_threshold : float (0-1)
        Minimum % of users in a role that must have a permission 
        for it to be included in the role definition
        Example: 0.8 means 80% of users must have the permission
    """
    
    def __init__(self, similarity_threshold=0.7, grouping_threshold=3, impact_threshold=0.8):
        """
        Parameters:
        -----------
        similarity_threshold : float
            User-to-user similarity threshold (0-1)
        grouping_threshold : int
            Minimum users per role
        impact_threshold : float
            Permission inclusion threshold (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.grouping_threshold = grouping_threshold
        self.impact_threshold = impact_threshold
        
        self.data = None
        self.labels = None
        self.roles = None
        self.metrics = None
        
        self._validate_thresholds()
    
    def _validate_thresholds(self):
        """Validate threshold parameters"""
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.grouping_threshold < 1:
            raise ValueError("grouping_threshold must be >= 1")
        if not 0 <= self.impact_threshold <= 1:
            raise ValueError("impact_threshold must be between 0 and 1")
    
    def load_data(self, filepath_or_dataframe):
        """Load user-permission matrix"""
        if isinstance(filepath_or_dataframe, pd.DataFrame):
            self.data = filepath_or_dataframe
        else:
            if filepath_or_dataframe.endswith('.csv'):
                self.data = pd.read_csv(filepath_or_dataframe, index_col=0)
            elif filepath_or_dataframe.endswith('.xlsx'):
                self.data = pd.read_excel(filepath_or_dataframe, index_col=0)
            else:
                raise ValueError("Unsupported format. Use CSV or XLSX")
        
        # Ensure binary
        if not self.data.isin([0, 1]).all().all():
            print("  Converting to binary (threshold=0.5)")
            self.data = (self.data > 0.5).astype(int)
        
        # Ensure string indices
        if not isinstance(self.data.index[0], str):
            self.data.index = [f"User_{i}" for i in range(len(self.data))]
        
        print(f" Loaded: {self.data.shape[0]} users × {self.data.shape[1]} permissions")
        print(f"   Sparsity: {(1 - self.data.sum().sum() / self.data.size) * 100:.1f}%")
        print(f"\n  Thresholds:")
        print(f"   • Similarity: {self.similarity_threshold} (user-user similarity)")
        print(f"   • Grouping: {self.grouping_threshold} (min users per role)")
        print(f"   • Impact: {self.impact_threshold} (permission inclusion)")
        
        return self.data
    
    @abstractmethod
    def tune_parameters(self):
        """Auto-tune algorithm parameters"""
        pass
    
    @abstractmethod
    def fit(self, **params):
        """Fit the clustering model"""
        pass
    
    def extract_roles(self):
        """
        Extract role definitions from cluster labels
        Applies grouping_threshold and impact_threshold
        """
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        roles = {}
        rejected_clusters = []
        
        for label in set(self.labels):
            if label == -1:  # Skip noise
                continue
            
            # Users in this cluster
            cluster_mask = self.labels == label
            cluster_users = self.data[cluster_mask]
            cluster_size = len(cluster_users)
            
            # Apply GROUPING THRESHOLD
            if cluster_size < self.grouping_threshold:
                rejected_clusters.append({
                    'label': label,
                    'size': cluster_size,
                    'reason': f'Below grouping threshold ({self.grouping_threshold})'
                })
                # Mark these users as noise
                self.labels[cluster_mask] = -1
                continue
            
            # Calculate permission frequencies
            permission_frequency = cluster_users.mean(axis=0)
            
            # Apply IMPACT THRESHOLD
            role_permissions = permission_frequency[
                permission_frequency >= self.impact_threshold
            ].index.tolist()
            
            # Calculate cohesion (average similarity within role)
            cohesion = float(permission_frequency[role_permissions].mean()) if role_permissions else 0.0
            
            # Calculate coverage (% of original permissions kept)
            original_perms = (permission_frequency > 0).sum()
            coverage = len(role_permissions) / original_perms if original_perms > 0 else 0.0
            
            roles[f"Role_{label}"] = {
                'permissions': role_permissions,
                'users': cluster_users.index.tolist(),
                'size': cluster_size,
                'permission_count': len(role_permissions),
                'cohesion': cohesion,
                'coverage': coverage,
                'permission_frequencies': permission_frequency[role_permissions].to_dict()
            }
        
        if rejected_clusters:
            print(f"\n  {len(rejected_clusters)} clusters rejected (grouping threshold):")
            for rc in rejected_clusters[:5]:  # Show first 5
                print(f"   • Cluster {rc['label']}: {rc['size']} users - {rc['reason']}")
            if len(rejected_clusters) > 5:
                print(f"   ... and {len(rejected_clusters)-5} more")
        
        self.roles = roles
        print(f"\n Extracted {len(roles)} valid roles")
        
        return roles
    
    def adjust_thresholds(self, similarity_threshold=None, grouping_threshold=None, 
                         impact_threshold=None):
        """
        Re-extract roles with different thresholds without re-clustering
        
        This is useful for fine-tuning without re-running the expensive clustering
        """
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        print(f"\n Adjusting thresholds...")
        
        if similarity_threshold is not None:
            print(f"   Similarity: {self.similarity_threshold} → {similarity_threshold}")
            self.similarity_threshold = similarity_threshold
        
        if grouping_threshold is not None:
            print(f"   Grouping: {self.grouping_threshold} → {grouping_threshold}")
            self.grouping_threshold = grouping_threshold
        
        if impact_threshold is not None:
            print(f"   Impact: {self.impact_threshold} → {impact_threshold}")
            self.impact_threshold = impact_threshold
        
        self._validate_thresholds()
        
        # Re-extract roles with new thresholds
        self.extract_roles()
        
        # Re-evaluate
        self.evaluate()
        
        return self.roles
    
    def evaluate(self):
        """Evaluate clustering quality"""
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.data.values
        labels = self.labels
        
        # Filter noise
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]
        
        metrics = {}
        
        # Silhouette score
        if len(set(labels_clean)) > 1 and len(X_clean) > 1:
            metrics['silhouette_score'] = float(silhouette_score(X_clean, labels_clean))
        else:
            metrics['silhouette_score'] = 0.0
        
        # Davies-Bouldin index
        if len(set(labels_clean)) > 1:
            metrics['davies_bouldin_index'] = float(davies_bouldin_score(X_clean, labels_clean))
        else:
            metrics['davies_bouldin_index'] = float('inf')
        
        # Basic stats
        metrics['num_roles'] = len(self.roles) if self.roles else 0
        metrics['num_noise'] = int(list(labels).count(-1))
        metrics['noise_percentage'] = float((list(labels).count(-1) / len(labels)) * 100)
        metrics['user_coverage'] = float(((len(labels) - list(labels).count(-1)) / len(labels)) * 100)
        
        # Role-specific metrics
        if self.roles:
            role_sizes = [r['size'] for r in self.roles.values()]
            metrics['avg_role_size'] = float(np.mean(role_sizes))
            metrics['std_role_size'] = float(np.std(role_sizes))
            metrics['min_role_size'] = int(np.min(role_sizes))
            metrics['max_role_size'] = int(np.max(role_sizes))
            
            perm_counts = [r['permission_count'] for r in self.roles.values()]
            metrics['avg_permissions_per_role'] = float(np.mean(perm_counts))
            metrics['min_permissions_per_role'] = int(np.min(perm_counts))
            metrics['max_permissions_per_role'] = int(np.max(perm_counts))
            
            cohesions = [r['cohesion'] for r in self.roles.values()]
            metrics['avg_cohesion'] = float(np.mean(cohesions))
            metrics['min_cohesion'] = float(np.min(cohesions))
            
            coverages = [r['coverage'] for r in self.roles.values()]
            metrics['avg_coverage'] = float(np.mean(coverages))
        
        # Threshold application metrics
        metrics['thresholds'] = {
            'similarity': self.similarity_threshold,
            'grouping': self.grouping_threshold,
            'impact': self.impact_threshold
        }
        
        self.metrics = metrics
        return metrics
    
    def visualize_results(self, method='tsne', save_path=None):
        """Visualize clusters in 2D"""
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        X = self.data.values
        labels = self.labels
        
        print(f" Reducing dimensions with {method.upper()}...")
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        else:
            reducer = PCA(n_components=2)
        
        X_reduced = reducer.fit_transform(X)
        
        # Plot
        plt.figure(figsize=(14, 9))
        
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'black'
                marker = 'x'
                label_name = 'Noise/Rejected'
                alpha = 0.3
                size = 80
            else:
                marker = 'o'
                label_name = f'Role {label}'
                alpha = 0.7
                size = 100
            
            mask = labels == label
            plt.scatter(
                X_reduced[mask, 0],
                X_reduced[mask, 1],
                c=[color],
                label=label_name,
                marker=marker,
                s=size,
                alpha=alpha,
                edgecolors='black',
                linewidth=0.5
            )
        
        plt.title(f'{self.__class__.__name__} - Role Clustering Visualization\n'
                 f'Similarity={self.similarity_threshold}, Grouping={self.grouping_threshold}, '
                 f'Impact={self.impact_threshold}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_roles_dataframe(self):
        """Get roles as a DataFrame"""
        if self.roles is None:
            raise ValueError("No roles extracted. Call extract_roles() first.")
        
        roles_data = []
        for role_name, role_info in sorted(self.roles.items(), 
                                          key=lambda x: x[1]['size'], 
                                          reverse=True):
            roles_data.append({
                'Role': role_name,
                'Users': role_info['size'],
                'Permissions': role_info['permission_count'],
                'Cohesion': f"{role_info['cohesion']:.2%}",
                'Coverage': f"{role_info['coverage']:.2%}",
                'Sample_Users': ', '.join(str(u) for u in role_info['users'][:3]) + 
                               (f' (+{role_info["size"]-3})' if role_info['size'] > 3 else ''),
                'Sample_Perms': ', '.join(role_info['permissions'][:3]) + 
                               (f' (+{role_info["permission_count"]-3})' if role_info['permission_count'] > 3 else '')
            })
        
        return pd.DataFrame(roles_data)
    
    def print_summary(self):
        """Print a nice summary of results"""
        if self.roles is None or self.metrics is None:
            print(" No results to display")
            return
        
        print("\n" + "="*80)
        print(f" {self.__class__.__name__} RESULTS")
        print("="*80)
        
        print(f"\n  Applied Thresholds:")
        print(f"   • Similarity Threshold: {self.similarity_threshold} (user-user grouping)")
        print(f"   • Grouping Threshold: {self.grouping_threshold} users (min role size)")
        print(f"   • Impact Threshold: {self.impact_threshold} (permission inclusion)")
        
        print(f"\n Clustering Metrics:")
        print(f"   • Number of roles: {self.metrics['num_roles']}")
        print(f"   • Noise/rejected users: {self.metrics['num_noise']} ({self.metrics['noise_percentage']:.1f}%)")
        print(f"   • User coverage: {self.metrics['user_coverage']:.1f}%")
        print(f"   • Silhouette score: {self.metrics['silhouette_score']:.3f}")
        print(f"   • Davies-Bouldin index: {self.metrics['davies_bouldin_index']:.3f}")
        
        print(f"\n Role Statistics:")
        print(f"   • Avg role size: {self.metrics.get('avg_role_size', 0):.1f} users")
        print(f"   • Role size range: {self.metrics.get('min_role_size', 0)}-{self.metrics.get('max_role_size', 0)} users")
        print(f"   • Avg permissions/role: {self.metrics.get('avg_permissions_per_role', 0):.1f}")
        print(f"   • Permission range: {self.metrics.get('min_permissions_per_role', 0)}-{self.metrics.get('max_permissions_per_role', 0)}")
        print(f"   • Avg cohesion: {self.metrics.get('avg_cohesion', 0):.2%}")
        print(f"   • Avg coverage: {self.metrics.get('avg_coverage', 0):.2%}")
        
        print(f"\n Role Details:")
        print(self.get_roles_dataframe().to_string(index=False))
    
    def export_results(self, output_dir='results'):
        """Export results to files"""
        import json
        from pathlib import Path
        
        Path(output_dir).mkdir(exist_ok=True)
        
        algo_name = self.__class__.__name__.lower()
        
        # Export roles
        roles_file = f"{output_dir}/{algo_name}_roles.json"
        with open(roles_file, 'w') as f:
            json.dump(self.roles, f, indent=2)
        print(f" Roles → {roles_file}")
        
        # Export metrics
        metrics_file = f"{output_dir}/{algo_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f" Metrics → {metrics_file}")
        
        # Export assignments
        assignments = []
        for role_name, role_info in self.roles.items():
            for user in role_info['users']:
                assignments.append({'User': user, 'Role': role_name})
        
        assignments_file = f"{output_dir}/{algo_name}_assignments.csv"
        pd.DataFrame(assignments).to_csv(assignments_file, index=False)
        print(f" Assignments → {assignments_file}")
        
        # Export detailed role definitions
        role_details = []
        for role_name, role_info in self.roles.items():
            for perm in role_info['permissions']:
                freq = role_info['permission_frequencies'].get(perm, 0)
                role_details.append({
                    'Role': role_name,
                    'Permission': perm,
                    'Frequency': freq,
                    'Meets_Impact_Threshold': freq >= self.impact_threshold
                })
        
        details_file = f"{output_dir}/{algo_name}_role_details.csv"
        pd.DataFrame(role_details).to_csv(details_file, index=False)
        print(f" Role details → {details_file}")