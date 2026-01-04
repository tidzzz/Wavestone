# role_mining_cli.py

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path

## the following two lines can be removed for windows users (wayland-matplotlib comptability)
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class RoleMiningCLI:
    """
    Interactive CLI for Role Mining with multiple algorithms
    """
    
    def __init__(self):
        self.data = None
        self.model = None
        self.results = {}
        
    def load_data(self, filepath):
        """Load user-permission matrix"""
        print(f"📂 Loading data from: {filepath}")
        
        # Support CSV, Excel, JSON
        if filepath.endswith('.csv'):
            self.data = pd.read_csv(filepath, index_col=0)
        elif filepath.endswith('.xlsx'):
            self.data = pd.read_excel(filepath, index_col=0)
        elif filepath.endswith('.json'):
            self.data = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV, XLSX, or JSON")
        
        # Ensure binary data
        if not self.data.isin([0, 1]).all().all():
            print("⚠️  Warning: Non-binary values detected. Converting to binary (threshold=0.5)")
            self.data = (self.data > 0.5).astype(int)
        
        print(f"✅ Loaded: {self.data.shape[0]} users × {self.data.shape[1]} permissions")
        print(f"   Sparsity: {(1 - self.data.sum().sum() / self.data.size) * 100:.1f}%")
        
        return self.data
    
    def suggest_dbscan_params(self):
        """
        Automatically suggest good DBSCAN parameters using k-distance graph
        """
        print("\n🔍 Analyzing data to suggest DBSCAN parameters...")
        
        X = self.data.values
        
        # Calculate k-distance for k = min_samples
        k_values = [3, 5, 10]
        suggested_params = []
        
        for k in k_values:
            if k >= len(X):
                continue
                
            # Find k-nearest neighbors
            neighbors = NearestNeighbors(n_neighbors=k, metric='jaccard')
            neighbors.fit(X)
            distances, indices = neighbors.kneighbors(X)
            
            # k-distance = distance to k-th nearest neighbor
            k_distances = np.sort(distances[:, -1])
            
            # Find "elbow" in k-distance plot (heuristic)
            # Use 95th percentile as suggested eps
            suggested_eps = np.percentile(k_distances, 95)
            
            suggested_params.append({
                'min_samples': k,
                'eps': round(suggested_eps, 3),
                'reasoning': f"95th percentile of {k}-distances"
            })
        
        print("\n💡 Suggested DBSCAN parameters:")
        for i, params in enumerate(suggested_params, 1):
            print(f"   Option {i}: eps={params['eps']}, min_samples={params['min_samples']}")
            print(f"              ({params['reasoning']})")
        
        # Plot k-distance graph for visualization
        self._plot_k_distance(X, k=5)
        
        return suggested_params
    
    def _plot_k_distance(self, X, k=5):
        """Plot k-distance graph to help choose eps"""
        neighbors = NearestNeighbors(n_neighbors=k, metric='jaccard')
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        k_distances = np.sort(distances[:, -1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances, linewidth=2)
        plt.xlabel('Points sorted by distance', fontsize=12)
        plt.ylabel(f'{k}-distance', fontsize=12)
        plt.title(f'K-distance Graph (k={k})\nLook for the "elbow" to choose eps', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add reference lines
        percentiles = [90, 95, 99]
        colors = ['green', 'orange', 'red']
        for p, c in zip(percentiles, colors):
            val = np.percentile(k_distances, p)
            plt.axhline(y=val, color=c, linestyle='--', alpha=0.7, 
                       label=f'{p}th percentile: {val:.3f}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('k_distance_plot.png', dpi=150)
        print(f"   📊 K-distance plot saved: k_distance_plot.png")
        plt.close()
    
    def run_dbscan(self, eps=None, min_samples=None, metric='jaccard', 
                   similarity_threshold=0.8):
        """
        Run DBSCAN clustering
        
        similarity_threshold: Minimum % of users in cluster that must have 
                            a permission for it to be included in role
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Auto-suggest if not provided
        if eps is None or min_samples is None:
            suggestions = self.suggest_dbscan_params()
            if not eps:
                eps = suggestions[1]['eps']  # Use middle suggestion
            if not min_samples:
                min_samples = suggestions[1]['min_samples']
        
        print(f"\n🚀 Running DBSCAN...")
        print(f"   Parameters: eps={eps}, min_samples={min_samples}, metric={metric}")
        print(f"   Similarity threshold: {similarity_threshold}")
        
        X = self.data.values
        
        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
        labels = dbscan.fit_predict(X)
        
        # Extract results
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"\n✅ Clustering complete:")
        print(f"   - {n_clusters} roles identified")
        print(f"   - {n_noise} noise users ({n_noise/len(labels)*100:.1f}%)")
        
        # Extract roles
        roles = self._extract_roles_from_labels(labels, similarity_threshold)
        
        # Evaluate
        metrics = self._evaluate_clustering(X, labels)
        
        self.results = {
            'algorithm': 'DBSCAN',
            'params': {'eps': eps, 'min_samples': min_samples, 'metric': metric,
                      'similarity_threshold': similarity_threshold},
            'labels': labels,
            'roles': roles,
            'metrics': metrics,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
        
        return self.results
    
    def run_kmeans(self, n_clusters=None, similarity_threshold=0.8):
        """Run K-means clustering"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Auto-suggest number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._suggest_n_clusters()
        
        print(f"\n🚀 Running K-means with {n_clusters} clusters...")
        print(f"   Similarity threshold: {similarity_threshold}")
        
        X = self.data.values
        
        # Run K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        print(f"\n✅ Clustering complete: {n_clusters} roles")
        
        # Extract roles
        roles = self._extract_roles_from_labels(labels, similarity_threshold)
        
        # Evaluate
        metrics = self._evaluate_clustering(X, labels)
        
        self.results = {
            'algorithm': 'K-means',
            'params': {'n_clusters': n_clusters, 'similarity_threshold': similarity_threshold},
            'labels': labels,
            'roles': roles,
            'metrics': metrics,
            'n_clusters': n_clusters,
            'n_noise': 0
        }
        
        return self.results
    
    def run_hierarchical(self, n_clusters=None, linkage='ward', 
                        similarity_threshold=0.8):
        """Run Hierarchical clustering"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if n_clusters is None:
            n_clusters = self._suggest_n_clusters()
        
        print(f"\n🚀 Running Hierarchical clustering...")
        print(f"   Parameters: {n_clusters} clusters, linkage={linkage}")
        print(f"   Similarity threshold: {similarity_threshold}")
        
        X = self.data.values
        
        # Run Hierarchical
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=linkage
        )
        labels = hierarchical.fit_predict(X)
        
        print(f"\n✅ Clustering complete: {n_clusters} roles")
        
        # Extract roles
        roles = self._extract_roles_from_labels(labels, similarity_threshold)
        
        # Evaluate
        metrics = self._evaluate_clustering(X, labels)
        
        self.results = {
            'algorithm': 'Hierarchical',
            'params': {'n_clusters': n_clusters, 'linkage': linkage,
                      'similarity_threshold': similarity_threshold},
            'labels': labels,
            'roles': roles,
            'metrics': metrics,
            'n_clusters': n_clusters,
            'n_noise': 0
        }
        
        return self.results
    
    def _suggest_n_clusters(self):
        """Suggest optimal number of clusters using elbow method"""
        print("\n🔍 Analyzing data to suggest number of clusters...")
        
        X = self.data.values
        inertias = []
        silhouettes = []
        k_range = range(2, min(20, len(X) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
        
        # Find elbow using second derivative
        inertias_normalized = (inertias - np.min(inertias)) / (np.max(inertias) - np.min(inertias))
        second_derivative = np.diff(inertias_normalized, 2)
        elbow_idx = np.argmax(second_derivative) + 2
        suggested_k = list(k_range)[elbow_idx]
        
        print(f"💡 Suggested number of clusters: {suggested_k}")
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=suggested_k, color='red', linestyle='--', 
                   label=f'Suggested k={suggested_k}')
        ax1.set_xlabel('Number of Clusters', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=suggested_k, color='red', linestyle='--', 
                   label=f'Suggested k={suggested_k}')
        ax2.set_xlabel('Number of Clusters', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cluster_selection.png', dpi=150)
        print(f"   📊 Analysis plot saved: cluster_selection.png")
        plt.close()
        
        return suggested_k
    
    def _extract_roles_from_labels(self, labels, similarity_threshold):
        """Extract role definitions from cluster labels"""
        roles = {}
        
        for label in set(labels):
            if label == -1:  # Skip noise for now
                continue
            
            cluster_mask = labels == label
            cluster_users = self.data[cluster_mask]
            
            # Permissions that appear in >= similarity_threshold% of users
            permission_frequency = cluster_users.mean(axis=0)
            role_permissions = permission_frequency[
                permission_frequency >= similarity_threshold
            ].index.tolist()
            
            roles[f"Role_{label}"] = {
                'permissions': role_permissions,
                'users': cluster_users.index.tolist(),
                'size': len(cluster_users),
                'permission_count': len(role_permissions),
                'cohesion': float(permission_frequency[role_permissions].mean()) if role_permissions else 0.0
            }
        
        return roles
    
    def _evaluate_clustering(self, X, labels):
        """Evaluate clustering quality"""
        metrics = {}
        
        # Filter noise
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]
        
        # Silhouette score
        if len(set(labels_clean)) > 1 and len(X_clean) > 1:
            metrics['silhouette_score'] = silhouette_score(X_clean, labels_clean)
        else:
            metrics['silhouette_score'] = 0.0
        
        # Davies-Bouldin index
        if len(set(labels_clean)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
        else:
            metrics['davies_bouldin'] = float('inf')
        
        # Custom metrics
        metrics['num_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        metrics['noise_percentage'] = (list(labels).count(-1) / len(labels)) * 100
        
        return metrics
    
    def compare_algorithms(self, algorithms=['dbscan', 'kmeans', 'hierarchical'],
                          similarity_threshold=0.8):
        """Compare multiple algorithms"""
        print("\n" + "="*60)
        print("🔬 ALGORITHM COMPARISON")
        print("="*60)
        
        comparison_results = []
        
        for algo in algorithms:
            print(f"\n{'─'*60}")
            if algo == 'dbscan':
                result = self.run_dbscan(similarity_threshold=similarity_threshold)
            elif algo == 'kmeans':
                result = self.run_kmeans(similarity_threshold=similarity_threshold)
            elif algo == 'hierarchical':
                result = self.run_hierarchical(similarity_threshold=similarity_threshold)
            
            comparison_results.append({
                'Algorithm': result['algorithm'],
                'Num Roles': result['n_clusters'],
                'Noise %': result['n_noise'] / len(self.data) * 100 if 'n_noise' in result else 0,
                'Silhouette': result['metrics']['silhouette_score'],
                'Davies-Bouldin': result['metrics']['davies_bouldin']
            })
        
        # Display comparison table
        print("\n" + "="*60)
        print("📊 COMPARISON RESULTS")
        print("="*60)
        df_comparison = pd.DataFrame(comparison_results)
        print(df_comparison.to_string(index=False))
        
        return comparison_results
    
    def display_results(self):
        """Display results in a nice format"""
        if not self.results:
            print("❌ No results to display. Run a clustering algorithm first.")
            return
        
        print("\n" + "="*60)
        print(f"📊 RESULTS: {self.results['algorithm']}")
        print("="*60)
        
        print(f"\n📈 Metrics:")
        for metric, value in self.results['metrics'].items():
            print(f"   {metric}: {value:.3f}")
        
        print(f"\n🎭 Roles Generated: {self.results['n_clusters']}")
        
        # Role details
        roles_data = []
        for role_name, role_info in self.results['roles'].items():
            roles_data.append({
                'Role': role_name,
                'Users': role_info['size'],
                'Permissions': role_info['permission_count'],
                'Cohesion': f"{role_info['cohesion']:.2f}",
                'Sample Users': ', '.join(str(u) for u in role_info['users'][:3]) + '...'
            })
        
        if roles_data:
            df_roles = pd.DataFrame(roles_data)
            print("\n" + df_roles.to_string(index=False))
        
        # Noise handling
        if self.results['n_noise'] > 0:
            print(f"\n⚠️  {self.results['n_noise']} users classified as noise")
    
    def export_results(self, output_path='results'):
        """Export results to files"""
        if not self.results:
            print("❌ No results to export.")
            return
        
        Path(output_path).mkdir(exist_ok=True)
        
        # Export roles to JSON
        roles_file = f"{output_path}/roles_{self.results['algorithm'].lower()}.json"
        with open(roles_file, 'w') as f:
            json.dump(self.results['roles'], f, indent=2)
        print(f"✅ Roles exported to: {roles_file}")
        
        # Export metrics
        metrics_file = f"{output_path}/metrics_{self.results['algorithm'].lower()}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        print(f"✅ Metrics exported to: {metrics_file}")
        
        # Export user-role assignments
        assignments = []
        for role_name, role_info in self.results['roles'].items():
            for user in role_info['users']:
                assignments.append({'User': user, 'Role': role_name})
        
        df_assignments = pd.DataFrame(assignments)
        assignments_file = f"{output_path}/user_role_assignments.csv"
        df_assignments.to_csv(assignments_file, index=False)
        print(f"✅ Assignments exported to: {assignments_file}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='🎭 Role Mining CLI - Automated Access Control Role Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first use)
  python role_mining_cli.py --data users_perms.csv --interactive
  
  # DBSCAN with auto-tuning
  python role_mining_cli.py --data users_perms.csv --algorithm dbscan
  
  # DBSCAN with custom parameters
  python role_mining_cli.py --data users_perms.csv --algorithm dbscan --eps 0.4 --min-samples 5
  
  # K-means with specific number of roles
  python role_mining_cli.py --data users_perms.csv --algorithm kmeans --n-clusters 8
  
  # Compare all algorithms
  python role_mining_cli.py --data users_perms.csv --compare
  
  # Adjust similarity threshold
  python role_mining_cli.py --data users_perms.csv --algorithm dbscan --threshold 0.9
        """
    )
    
    # Required arguments
    parser.add_argument('--data', required=True, 
                       help='Path to user-permission matrix (CSV, XLSX, or JSON)')
    
    # Algorithm selection
    parser.add_argument('--algorithm', choices=['dbscan', 'kmeans', 'hierarchical'],
                       help='Clustering algorithm to use')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare all algorithms')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode with prompts')
    
    # DBSCAN parameters
    parser.add_argument('--eps', type=float,
                       help='DBSCAN epsilon (auto-suggested if not provided)')
    
    parser.add_argument('--min-samples', type=int,
                       help='DBSCAN min_samples (auto-suggested if not provided)')
    
    parser.add_argument('--metric', default='jaccard',
                       choices=['jaccard', 'cosine', 'euclidean'],
                       help='Distance metric for DBSCAN')
    
    # K-means/Hierarchical parameters
    parser.add_argument('--n-clusters', type=int,
                       help='Number of clusters for K-means/Hierarchical (auto-suggested if not provided)')
    
    parser.add_argument('--linkage', default='ward',
                       choices=['ward', 'complete', 'average', 'single'],
                       help='Linkage method for Hierarchical clustering')
    
    # General parameters
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='Similarity threshold (0-1) for including permissions in roles (default: 0.8)')
    
    parser.add_argument('--output', default='results',
                       help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = RoleMiningCLI()
    
    # Load data
    try:
        cli.load_data(args.data)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("🎯 INTERACTIVE MODE")
        print("="*60)
        
        print("\nAvailable algorithms:")
        print("  1. DBSCAN (auto-detects number of roles, handles noise)")
        print("  2. K-means (requires number of roles)")
        print("  3. Hierarchical (requires number of roles, shows hierarchy)")
        print("  4. Compare all")
        
        choice = input("\nSelect algorithm (1-4): ").strip()
        
        if choice == '1':
            args.algorithm = 'dbscan'
            if not args.eps:
                suggestions = cli.suggest_dbscan_params()
                eps_choice = input(f"\nChoose eps (press Enter for suggested {suggestions[1]['eps']}): ").strip()
                args.eps = float(eps_choice) if eps_choice else suggestions[1]['eps']
            if not args.min_samples:
                min_samples_choice = input(f"Choose min_samples (press Enter for suggested {suggestions[1]['min_samples']}): ").strip()
                args.min_samples = int(min_samples_choice) if min_samples_choice else suggestions[1]['min_samples']
        
        elif choice == '2':
            args.algorithm = 'kmeans'
        
        elif choice == '3':
            args.algorithm = 'hierarchical'
        
        elif choice == '4':
            args.compare = True
        
        threshold_input = input(f"\nSimilarity threshold (press Enter for {args.threshold}): ").strip()
        if threshold_input:
            args.threshold = float(threshold_input)
    
    # Run clustering
    if args.compare:
        cli.compare_algorithms(similarity_threshold=args.threshold)
    elif args.algorithm == 'dbscan':
        cli.run_dbscan(
            eps=args.eps,
            min_samples=args.min_samples,
            metric=args.metric,
            similarity_threshold=args.threshold
        )
    elif args.algorithm == 'kmeans':
        cli.run_kmeans(
            n_clusters=args.n_clusters,
            similarity_threshold=args.threshold
        )
    elif args.algorithm == 'hierarchical':
        cli.run_hierarchical(
            n_clusters=args.n_clusters,
            linkage=args.linkage,
            similarity_threshold=args.threshold
        )
    else:
        print("❌ Please specify --algorithm or use --compare")
        return
    
    # Display and export results
    cli.display_results()
    cli.export_results(args.output)
    
    print(f"\n✅ Done! Results saved to: {args.output}/")


if __name__ == "__main__":
    main()