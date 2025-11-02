# gnn_miner_optimized.py
"""
Optimized Graph Neural Network Role Miner
Focus: Maximum quality over speed
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

from base_miner import BaseRoleMiner

# Optional dependencies
try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("  Gensim not installed. Install with: pip install gensim")

try:
    from node2vec import Node2Vec as N2V
    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False
    print("  Node2Vec not installed. Install with: pip install node2vec")

try:
    from sklearn.manifold import TSNE
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class GNNRoleMiner(BaseRoleMiner):
    """
    High-Performance Graph Neural Network for Role Mining
    
    Focus on quality over speed:
    - Multiple embedding methods with ensemble
    - Advanced clustering with refinement
    - Iterative optimization
    - Quality validation at each step
    """
    
    def __init__(self, 
                 # Embedding parameters
                 embedding_dim=128,  # Increased for better representation
                 embedding_method='ensemble',  # 'svd', 'node2vec', 'deepwalk', 'nmf', 'ensemble'
                 
                 # Graph walk parameters (for Node2Vec/DeepWalk)
                 walk_length=80,
                 num_walks=100,
                 p=1.0,  # Return parameter (Node2Vec)
                 q=1.0,  # In-out parameter (Node2Vec)
                 window_size=10,
                 epochs=10,
                 
                 # Clustering parameters
                 clustering_method='spectral',  # 'kmeans', 'spectral', 'hierarchical', 'auto'
                 refinement_iterations=3,  # Post-clustering refinement
                 
                 # Quality parameters
                 min_cluster_quality=0.6,  # Minimum silhouette per cluster
                 merge_similar_roles=True,  # Merge very similar roles
                 role_similarity_threshold=0.85,  # For merging
                 
                 **kwargs):
        """
        Parameters:
        -----------
        embedding_dim : int
            Dimensionality of embeddings (higher = more expressive)
        embedding_method : str
            'svd', 'node2vec', 'deepwalk', 'nmf', 'ensemble'
            ensemble = combine multiple methods for best quality
        clustering_method : str
            'kmeans', 'spectral', 'hierarchical', 'auto'
        refinement_iterations : int
            Number of refinement passes
        min_cluster_quality : float
            Minimum acceptable silhouette score per cluster
        """
        super().__init__(**kwargs)
        
        # Embedding config
        self.embedding_dim = embedding_dim
        self.embedding_method = embedding_method
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.epochs = epochs
        
        # Clustering config
        self.clustering_method = clustering_method
        self.refinement_iterations = refinement_iterations
        self.min_cluster_quality = min_cluster_quality
        self.merge_similar_roles = merge_similar_roles
        self.role_similarity_threshold = role_similarity_threshold
        
        # Internal state
        self.graph = None
        self.user_embeddings = None
        self.embedding_variance = None
        self.embedding_components = {}  # Store multiple embeddings
        self.cluster_quality_scores = {}
        self.suggested_n_clusters = None
    
    def _build_weighted_graph(self):
        """
        Build weighted bipartite graph with edge weights
        Weight = importance/rarity of permission
        """
        print("\n Building weighted bipartite graph...")
        
        self.graph = nx.Graph()
        
        n_users = len(self.data)
        n_perms = len(self.data.columns)
        
        # Add user nodes
        user_nodes = [f"user_{i}" for i in range(n_users)]
        self.graph.add_nodes_from(user_nodes, bipartite=0, node_type='user')
        
        # Add permission nodes
        perm_nodes = [f"perm_{col}" for col in self.data.columns]
        self.graph.add_nodes_from(perm_nodes, bipartite=1, node_type='permission')
        
        # Calculate permission weights (inverse frequency = rarity)
        perm_counts = self.data.sum(axis=0)
        perm_weights = {}
        
        for perm, count in perm_counts.items():
            if count > 0:
                # Rare permissions get higher weight
                perm_weights[perm] = np.log(n_users / (count + 1)) + 1
            else:
                perm_weights[perm] = 1.0
        
        # Add weighted edges (vectorized)
        edges_with_weights = []
        user_indices, perm_indices = np.where(self.data.values == 1)
        
        for u_idx, p_idx in zip(user_indices, perm_indices):
            perm_name = self.data.columns[p_idx]
            weight = perm_weights[perm_name]
            edges_with_weights.append((
                f"user_{u_idx}", 
                f"perm_{perm_name}", 
                weight
            ))
        
        self.graph.add_weighted_edges_from(edges_with_weights)
        
        # Statistics
        avg_weight = np.mean([e[2] for e in edges_with_weights])
        density = (2 * len(edges_with_weights)) / (n_users * n_perms)
        
        print(f"  ✓ Users: {n_users:,}")
        print(f"  ✓ Permissions: {n_perms:,}")
        print(f"  ✓ Weighted edges: {len(edges_with_weights):,}")
        print(f"  ✓ Avg edge weight: {avg_weight:.2f}")
        print(f"  ✓ Graph density: {density:.6f}")
        
        return self.graph
    
    def _compute_svd_embeddings(self, augmented=True):
        """
        Enhanced Truncated SVD with optional augmentation
        
        Parameters:
        -----------
        augmented : bool
            If True, augment with TF-IDF style weighting
        """
        print("   Computing SVD embeddings...")
        
        try:
            X = self.data.values.astype(float)
            
            # Optional: TF-IDF style weighting (boosts rare permissions)
            if augmented:
                # IDF weighting
                perm_counts = X.sum(axis=0)
                idf = np.log(len(X) / (perm_counts + 1)) + 1
                X_weighted = X * idf
            else:
                X_weighted = X
            
            # Compute SVD with optimal components
            max_components = min(min(X.shape) - 1, self.embedding_dim)
            
            svd = TruncatedSVD(n_components=max_components, 
                              random_state=42,
                              n_iter=10)  # More iterations for stability
            embeddings = svd.fit_transform(X_weighted)
            
            # Normalize embeddings
            embeddings = normalize(embeddings, norm='l2')
            
            variance = svd.explained_variance_ratio_.sum()
            
            print(f"    → Shape: {embeddings.shape}")
            print(f"    → Explained variance: {variance:.2%}")
            print(f"    → Augmented: {augmented}")
            
            return embeddings, variance
            
        except Exception as e:
            print(f"    ❌ SVD failed: {e}")
            return None, 0.0
    
    def _compute_node2vec_embeddings(self):
        """
        Node2Vec embeddings (best quality for graphs)
        """
        if not HAS_NODE2VEC:
            print("    Node2Vec not available")
            return None
        
        print("   Computing Node2Vec embeddings (high quality)...")
        
        try:
            # Configure Node2Vec
            node2vec = N2V(
                self.graph,
                dimensions=self.embedding_dim,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                p=self.p,
                q=self.q,
                workers=4,
                seed=42,
                quiet=True
            )
            
            print(f"    → Generating walks...")
            model = node2vec.fit(
                window=self.window_size,
                min_count=1,
                batch_words=4,
                epochs=self.epochs
            )
            
            # Extract user embeddings
            embeddings = []
            for i in range(len(self.data)):
                user_id = f"user_{i}"
                if user_id in model.wv:
                    embeddings.append(model.wv[user_id])
                else:
                    embeddings.append(np.zeros(self.embedding_dim))
            
            embeddings = np.array(embeddings)
            embeddings = normalize(embeddings, norm='l2')
            
            print(f"    → Shape: {embeddings.shape}")
            print(f"    → p={self.p}, q={self.q}")
            
            return embeddings
            
        except Exception as e:
            print(f"     Node2Vec failed: {e}")
            return None
    
    def _compute_deepwalk_embeddings(self):
        """
        DeepWalk embeddings (faster alternative to Node2Vec)
        """
        if not HAS_GENSIM:
            print("    DeepWalk not available (needs gensim)")
            return None
        
        print("  🚶 Computing DeepWalk embeddings...")
        
        try:
            # Get user nodes
            user_nodes = [n for n in self.graph.nodes() 
                         if self.graph.nodes[n]['node_type'] == 'user']
            
            # Generate random walks
            print(f"    → Generating {self.num_walks} walks of length {self.walk_length}...")
            walks = []
            
            for _ in range(self.num_walks):
                np.random.shuffle(user_nodes)  # Random order
                for node in user_nodes:
                    walk = self._biased_random_walk(node)
                    if len(walk) > 1:
                        walks.append(walk)
            
            print(f"    → Training Word2Vec on {len(walks):,} walks...")
            
            # Train Word2Vec
            model = Word2Vec(
                walks,
                vector_size=self.embedding_dim,
                window=self.window_size,
                min_count=1,
                sg=1,  # Skip-gram
                workers=4,
                epochs=self.epochs,
                seed=42
            )
            
            # Extract embeddings
            embeddings = []
            for i in range(len(self.data)):
                user_id = f"user_{i}"
                try:
                    embeddings.append(model.wv[user_id])
                except KeyError:
                    embeddings.append(np.zeros(self.embedding_dim))
            
            embeddings = np.array(embeddings)
            embeddings = normalize(embeddings, norm='l2')
            
            print(f"    → Shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            print(f"     DeepWalk failed: {e}")
            return None
    
    def _biased_random_walk(self, start_node, use_weights=True):
        """
        Biased random walk considering edge weights
        """
        walk = [start_node]
        current = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(current))
            
            if not neighbors:
                break
            
            if use_weights and self.graph.is_weighted():
                # Weighted sampling
                weights = [self.graph[current][n].get('weight', 1.0) 
                          for n in neighbors]
                weights = np.array(weights)
                probs = weights / weights.sum()
                current = np.random.choice(neighbors, p=probs)
            else:
                # Uniform sampling
                current = np.random.choice(neighbors)
            
            walk.append(current)
        
        return walk
    
    def _compute_nmf_embeddings(self):
        """
        Non-negative Matrix Factorization (interpretable)
        """
        print("   Computing NMF embeddings...")
        
        try:
            X = self.data.values.astype(float)
            
            # NMF requires non-negative data
            nmf = NMF(
                n_components=self.embedding_dim,
                init='nndsvda',  # Better initialization
                random_state=42,
                max_iter=400,
                alpha_W=0.1,  # L1 regularization
                alpha_H=0.1,
                l1_ratio=0.5
            )
            
            embeddings = nmf.fit_transform(X)
            
            # Quality metric
            reconstruction = nmf.inverse_transform(embeddings)
            mse = np.mean((X - reconstruction) ** 2)
            
            # Normalize
            embeddings = normalize(embeddings, norm='l2')
            
            print(f"    → Shape: {embeddings.shape}")
            print(f"    → Reconstruction MSE: {mse:.4f}")
            
            return embeddings
            
        except Exception as e:
            print(f"     NMF failed: {e}")
            return None
    
    def _compute_ensemble_embeddings(self):
        """
        Ensemble of multiple embedding methods for maximum quality
        """
        print("\n Computing ENSEMBLE embeddings (best quality)...")
        
        embeddings_list = []
        weights = []
        methods_used = []
        
        # 1. SVD (always works)
        print("\n1️ SVD component:")
        svd_emb, svd_var = self._compute_svd_embeddings(augmented=True)
        if svd_emb is not None:
            embeddings_list.append(svd_emb)
            weights.append(svd_var)  # Weight by variance explained
            methods_used.append('SVD')
            self.embedding_components['svd'] = svd_emb
        
        # 2. Node2Vec (best for graphs, if available)
        if HAS_NODE2VEC:
            print("\n2️ Node2Vec component:")
            n2v_emb = self._compute_node2vec_embeddings()
            if n2v_emb is not None:
                embeddings_list.append(n2v_emb)
                weights.append(1.5)  # Higher weight (usually best quality)
                methods_used.append('Node2Vec')
                self.embedding_components['node2vec'] = n2v_emb
        
        # 3. DeepWalk (good alternative)
        if HAS_GENSIM and not HAS_NODE2VEC:
            print("\n3️ DeepWalk component:")
            dw_emb = self._compute_deepwalk_embeddings()
            if dw_emb is not None:
                embeddings_list.append(dw_emb)
                weights.append(1.0)
                methods_used.append('DeepWalk')
                self.embedding_components['deepwalk'] = dw_emb
        
        # 4. NMF (adds interpretability)
        print("\n4️ NMF component:")
        nmf_emb = self._compute_nmf_embeddings()
        if nmf_emb is not None:
            embeddings_list.append(nmf_emb)
            weights.append(0.8)  # Lower weight (less discriminative)
            methods_used.append('NMF')
            self.embedding_components['nmf'] = nmf_emb
        
        if not embeddings_list:
            raise RuntimeError("All embedding methods failed!")
        
        # Combine embeddings
        print(f"\n Combining {len(embeddings_list)} embedding methods...")
        print(f"   Methods: {', '.join(methods_used)}")
        
        # Weighted concatenation
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Concatenate all embeddings
        combined = np.concatenate(embeddings_list, axis=1)
        
        # Optional: Reduce dimensionality if too large
        if combined.shape[1] > self.embedding_dim * 2:
            print(f"   Reducing from {combined.shape[1]} to {self.embedding_dim * 2} dims...")
            reducer = TruncatedSVD(n_components=self.embedding_dim * 2, random_state=42)
            combined = reducer.fit_transform(combined)
        
        # Final normalization
        combined = normalize(combined, norm='l2')
        
        print(f"   ✓ Final embedding shape: {combined.shape}")
        
        return combined
    
    def compute_embeddings(self, method=None):
        """
        Compute user embeddings with specified method
        
        This method can be called before fit() for consistency
        """
        if self.data is None:
            raise ValueError("Load data first with load_data()")
        
        print("\n" + "="*80)
        print(" COMPUTING USER EMBEDDINGS")
        print("="*80)
        
        # Build graph first
        if self.graph is None:
            self._build_weighted_graph()
        
        # Choose method
        if method is None:
            method = self.embedding_method
        
        print(f"\n Method: {method.upper()}")
        
        # Compute embeddings based on method
        if method == 'ensemble':
            embeddings = self._compute_ensemble_embeddings()
        elif method == 'svd':
            embeddings, var = self._compute_svd_embeddings(augmented=True)
            self.embedding_variance = var
        elif method == 'node2vec':
            embeddings = self._compute_node2vec_embeddings()
        elif method == 'deepwalk':
            embeddings = self._compute_deepwalk_embeddings()
        elif method == 'nmf':
            embeddings = self._compute_nmf_embeddings()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if embeddings is None:
            raise RuntimeError(f"Embedding computation failed for method: {method}")
        
        self.user_embeddings = embeddings
        
        print(f"\n Embeddings ready: {embeddings.shape}")
        print(f"   Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.3f}")
        print(f"   Std norm: {np.linalg.norm(embeddings, axis=1).std():.3f}")
        
        return embeddings
    
    def tune_parameters(self, max_clusters=30, methods=['silhouette', 'calinski', 'davies'],
                       plot=True, save_plot=None):
        """
        Advanced parameter tuning with multiple quality metrics
        
        Parameters:
        -----------
        max_clusters : int
            Maximum number of clusters to test
        methods : list
            Quality metrics to use ['silhouette', 'calinski', 'davies']
        """
        if self.user_embeddings is None:
            print("  Computing embeddings first...")
            self.compute_embeddings()
        
        print("\n" + "="*80)
        print(" ADVANCED PARAMETER TUNING")
        print("="*80)
        
        X = self.user_embeddings
        n_samples = len(X)
        
        # Determine range
        min_k = max(2, self.grouping_threshold)
        max_k = min(max_clusters, n_samples // self.grouping_threshold, 100)
        
        # Adaptive step based on range
        if max_k - min_k > 20:
            step = 2
        else:
            step = 1
        
        k_range = list(range(min_k, max_k + 1, step))
        
        print(f"\n Testing k from {min_k} to {max_k} (step={step})...")
        print(f"   Metrics: {', '.join(methods)}")
        
        # Compute metrics for each k
        results = {
            'k': [],
            'silhouette': [],
            'calinski': [],
            'davies': [],
            'combined_score': []
        }
        
        for k in k_range:
            try:
                # Cluster with current k
                if self.clustering_method == 'spectral' and k < 100:
                    clusterer = SpectralClustering(
                        n_clusters=k,
                        random_state=42,
                        affinity='nearest_neighbors',
                        n_neighbors=min(10, n_samples-1)
                    )
                    labels = clusterer.fit_predict(X)
                else:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                    labels = kmeans.fit_predict(X)
                
                # Compute metrics
                if len(set(labels)) > 1:
                    if 'silhouette' in methods:
                        sil = silhouette_score(X, labels)
                        results['silhouette'].append(sil)
                    else:
                        results['silhouette'].append(0)
                    
                    if 'calinski' in methods:
                        cal = calinski_harabasz_score(X, labels)
                        results['calinski'].append(cal)
                    else:
                        results['calinski'].append(0)
                    
                    if 'davies' in methods:
                        dav = davies_bouldin_score(X, labels)
                        results['davies'].append(dav)
                    else:
                        results['davies'].append(float('inf'))
                else:
                    results['silhouette'].append(0)
                    results['calinski'].append(0)
                    results['davies'].append(float('inf'))
                
                results['k'].append(k)
                
                # Combined score (higher is better)
                # Normalize metrics and combine
                combined = results['silhouette'][-1] * 100
                if results['davies'][-1] != float('inf'):
                    combined -= results['davies'][-1] * 10
                
                results['combined_score'].append(combined)
                
                if k % 5 == 0:
                    print(f"   k={k}: sil={results['silhouette'][-1]:.3f}, "
                          f"davies={results['davies'][-1]:.3f}")
                
            except Exception as e:
                print(f"     k={k} failed: {e}")
                results['k'].append(k)
                results['silhouette'].append(0)
                results['calinski'].append(0)
                results['davies'].append(float('inf'))
                results['combined_score'].append(-1000)
        
        # Find optimal k
        best_idx = np.argmax(results['combined_score'])
        best_k = results['k'][best_idx]
        
        print(f"\n OPTIMAL CONFIGURATION:")
        print(f"   k = {best_k}")
        print(f"   Silhouette: {results['silhouette'][best_idx]:.3f}")
        print(f"   Davies-Bouldin: {results['davies'][best_idx]:.3f}")
        print(f"   Calinski-Harabasz: {results['calinski'][best_idx]:.1f}")
        
        self.suggested_n_clusters = best_k
        
        # Plot results
        if plot:
            self._plot_advanced_tuning(results, best_k, save_plot)
        
        return best_k
    
    def _plot_advanced_tuning(self, results, best_k, save_path):
        """Plot multiple quality metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        k_vals = results['k']
        
        # Silhouette
        ax = axes[0, 0]
        ax.plot(k_vals, results['silhouette'], 'b-o', linewidth=2, markersize=6)
        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score (higher is better)')
        ax.grid(True, alpha=0.3)
        
        # Davies-Bouldin
        ax = axes[0, 1]
        davies_clean = [d if d != float('inf') else max([d for d in results['davies'] if d != float('inf')]) 
                       for d in results['davies']]
        ax.plot(k_vals, davies_clean, 'r-o', linewidth=2, markersize=6)
        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Davies-Bouldin Index')
        ax.set_title('Davies-Bouldin Index (lower is better)')
        ax.grid(True, alpha=0.3)
        
        # Calinski-Harabasz
        ax = axes[1, 0]
        ax.plot(k_vals, results['calinski'], 'g-o', linewidth=2, markersize=6)
        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Calinski-Harabasz Score')
        ax.set_title('Calinski-Harabasz Score (higher is better)')
        ax.grid(True, alpha=0.3)
        
        # Combined score
        ax = axes[1, 1]
        ax.plot(k_vals, results['combined_score'], 'purple', linewidth=2, marker='o', markersize=6)
        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Optimal: k={best_k}')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Combined Score')
        ax.set_title('Combined Quality Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Advanced Clustering Quality Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"\n Tuning plot → {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _cluster_with_method(self, n_clusters, method=None):
        """
        Cluster embeddings with specified method
        """
        if method is None:
            method = self.clustering_method
        
        X = self.user_embeddings
        
        if method == 'auto':
            # Auto-select based on dataset size
            if len(X) > 5000:
                method = 'kmeans'
            elif len(X) > 1000:
                method = 'spectral'
            else:
                method = 'spectral'
        
        print(f"  Clustering method: {method}")
        
        try:
            if method == 'kmeans':
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=20,  # More initializations
                    max_iter=500,
                    algorithm='elkan'  # Faster for large datasets
                )
                labels = clusterer.fit_predict(X)
                
            elif method == 'spectral':
                if n_clusters > 100:
                    print("      Spectral clustering slow for many clusters, using K-means")
                    return self._cluster_with_method(n_clusters, 'kmeans')
                
                clusterer = SpectralClustering(
                    n_clusters=n_clusters,
                    random_state=42,
                    affinity='nearest_neighbors',
                    n_neighbors=min(20, len(X)-1),
                    assign_labels='kmeans'
                )
                labels = clusterer.fit_predict(X)
                
            elif method == 'hierarchical':
                from sklearn.cluster import AgglomerativeClustering
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
                labels = clusterer.fit_predict(X)
            
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            return labels
            
        except Exception as e:
            print(f"     {method} failed: {e}")
            print(f"    → Falling back to K-means")
            return self._cluster_with_method(n_clusters, 'kmeans')
    
    def _refine_clusters(self, labels, iteration=1):
        """
        Iterative cluster refinement
        - Identify low-quality clusters
        - Reassign outliers
        - Merge similar clusters
        """
        print(f"\n   Refinement iteration {iteration}...")
        
        X = self.user_embeddings
        unique_labels = set(labels)
        
        # Calculate per-cluster quality
        cluster_qualities = {}
        
        for label in unique_labels:
            if label == -1:
                continue
            
            mask = labels == label
            cluster_points = X[mask]
            
            if len(cluster_points) < 2:
                cluster_qualities[label] = 0.0
                continue
            
            # Intra-cluster distances
            from scipy.spatial.distance import pdist
            distances = pdist(cluster_points, metric='euclidean')
            
            if len(distances) > 0:
                # Cohesion = average intra-cluster distance (lower is better)
                cohesion = np.mean(distances)
                
                # Convert to quality score (higher is better)
                quality = 1.0 / (1.0 + cohesion)
                cluster_qualities[label] = quality
            else:
                cluster_qualities[label] = 0.0
        
        # Identify low-quality clusters
        low_quality_clusters = [
            label for label, quality in cluster_qualities.items()
            if quality < self.min_cluster_quality
        ]
        
        if low_quality_clusters:
            print(f"    → Found {len(low_quality_clusters)} low-quality clusters")
            
            # Reassign points from low-quality clusters
            for bad_label in low_quality_clusters:
                mask = labels == bad_label
                bad_points = X[mask]
                
                if len(bad_points) == 0:
                    continue
                
                # Find nearest good cluster for each point
                good_labels = [l for l in unique_labels 
                              if l not in low_quality_clusters and l != -1]
                
                if not good_labels:
                    continue
                
                # Calculate centroids of good clusters
                centroids = []
                for good_label in good_labels:
                    good_mask = labels == good_label
                    centroid = X[good_mask].mean(axis=0)
                    centroids.append(centroid)
                
                centroids = np.array(centroids)
                
                # Reassign each bad point to nearest good cluster
                for i, point in enumerate(bad_points):
                    distances = np.linalg.norm(centroids - point, axis=1)
                    nearest_cluster_idx = np.argmin(distances)
                    
                    # Get the original index in the full dataset
                    original_idx = np.where(mask)[0][i]
                    labels[original_idx] = good_labels[nearest_cluster_idx]
        
        # Merge very similar clusters if requested
        if self.merge_similar_roles:
            labels = self._merge_similar_clusters(labels, X)
        
        return labels
    
    def _merge_similar_clusters(self, labels, X):
        """
        Merge clusters that are very similar
        """
        unique_labels = [l for l in set(labels) if l != -1]
        
        if len(unique_labels) < 2:
            return labels
        
        # Calculate cluster centroids
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = X[mask].mean(axis=0)
        
        # Calculate pairwise similarities
        merge_pairs = []
        
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                similarity = np.dot(centroids[label1], centroids[label2])
                
                if similarity > self.role_similarity_threshold:
                    merge_pairs.append((label1, label2, similarity))
        
        if merge_pairs:
            print(f"    → Merging {len(merge_pairs)} similar cluster pairs...")
            
            # Merge clusters
            merge_map = {}
            for label1, label2, sim in sorted(merge_pairs, key=lambda x: -x[2]):
                # Map both to the smaller label
                target = min(label1, label2)
                source = max(label1, label2)
                
                # Update existing mappings
                for key in list(merge_map.keys()):
                    if merge_map[key] == source:
                        merge_map[key] = target
                
                merge_map[source] = target
            
            # Apply merge map
            for i in range(len(labels)):
                if labels[i] in merge_map:
                    labels[i] = merge_map[labels[i]]
        
        return labels
    
    def fit(self, n_clusters=None, auto_tune=True, verbose=True):
        """
        Fit optimized GNN role miner with iterative refinement
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (auto-tuned if None)
        auto_tune : bool
            Perform advanced parameter tuning
        verbose : bool
            Print detailed progress
        """
        if self.data is None:
            raise ValueError("Load data first with load_data()")
        
        if verbose:
            print("\n" + "="*80)
            print(" FITTING OPTIMIZED GNN ROLE MINER")
            print("="*80)
            print(f"\nConfiguration:")
            print(f"  • Embedding method: {self.embedding_method}")
            print(f"  • Clustering method: {self.clustering_method}")
            print(f"  • Refinement iterations: {self.refinement_iterations}")
            print(f"  • Min cluster quality: {self.min_cluster_quality}")
        
        # Step 1: Compute embeddings
        if self.user_embeddings is None:
            self.compute_embeddings()
        
        # Step 2: Tune parameters
        if n_clusters is None:
            if auto_tune:
                if verbose:
                    print("\n" + "-"*80)
                n_clusters = self.tune_parameters(plot=verbose)
            else:
                # Heuristic
                n_clusters = max(5, len(self.data) // 100)
                if verbose:
                    print(f"\n  Using heuristic: n_clusters={n_clusters}")
        
        # Step 3: Initial clustering
        if verbose:
            print("\n" + "-"*80)
            print("  INITIAL CLUSTERING")
            print("-"*80)
        
        self.labels = self._cluster_with_method(n_clusters)
        
        initial_n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        if verbose:
            print(f"  ✓ Initial clusters: {initial_n_clusters}")
        
        # Step 4: Iterative refinement
        if self.refinement_iterations > 0 and verbose:
            print("\n" + "-"*80)
            print(" ITERATIVE REFINEMENT")
            print("-"*80)
        
        for iteration in range(1, self.refinement_iterations + 1):
            self.labels = self._refine_clusters(self.labels, iteration)
            
            refined_n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
            if verbose:
                print(f"    ✓ After iteration {iteration}: {refined_n_clusters} clusters")
        
        # Step 5: Extract roles with thresholds
        if verbose:
            print("\n" + "-"*80)
            print(" EXTRACTING ROLES")
            print("-"*80)
        
        self.extract_roles()
        
        # Step 6: Evaluate
        if verbose:
            print("\n" + "-"*80)
            print(" EVALUATING QUALITY")
            print("-"*80)
        
        self.evaluate()
        
        if verbose:
            final_n_roles = len(self.roles)
            print(f"\n OPTIMIZATION COMPLETE")
            print(f"   Initial clusters: {initial_n_clusters}")
            print(f"   Final valid roles: {final_n_roles}")
            print(f"   Noise users: {self.metrics['num_noise']}")
        
        return self
    
    def extract_roles(self):
        """
        Enhanced role extraction with validation
        """
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        roles = {}
        rejected_clusters = []
        valid_clusters = 0
        
        for label in set(self.labels):
            if label == -1:
                continue
            
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
                self.labels[cluster_mask] = -1
                continue
            
            # Calculate permission frequencies
            permission_frequency = cluster_users.mean(axis=0)
            
            # Apply IMPACT THRESHOLD
            role_permissions = permission_frequency[
                permission_frequency >= self.impact_threshold
            ].index.tolist()
            
            # Reject roles without permissions
            if len(role_permissions) == 0:
                rejected_clusters.append({
                    'label': label,
                    'size': cluster_size,
                    'reason': f'No permissions meet impact threshold ({self.impact_threshold})'
                })
                self.labels[cluster_mask] = -1
                continue
            
            # Calculate role quality metrics
            cohesion_values = permission_frequency[role_permissions]
            cohesion = float(cohesion_values.mean())
            
            # Coverage
            original_perms_mask = (cluster_users.sum(axis=0) > 0)
            original_perms_count = int(original_perms_mask.sum())
            coverage = float(len(role_permissions) / original_perms_count) if original_perms_count > 0 else 0.0
            
            # Permission frequencies (native Python types)
            permission_frequencies_dict = {
                perm: float(permission_frequency[perm])
                for perm in role_permissions
            }
            
            # Store role
            role_name = f"Role_{valid_clusters + 1}"
            roles[role_name] = {
                'permissions': role_permissions,
                'users': cluster_users.index.tolist(),
                'size': int(cluster_size),
                'permission_count': int(len(role_permissions)),
                'cohesion': float(cohesion),
                'coverage': float(coverage),
                'permission_frequencies': permission_frequencies_dict,
                'original_permissions_count': int(original_perms_count),
                'impact_threshold_applied': float(self.impact_threshold)
            }
            
            valid_clusters += 1
        
        if rejected_clusters:
            print(f"\n    {len(rejected_clusters)} clusters rejected:")
            for rc in rejected_clusters[:5]:
                print(f"     • Cluster {rc['label']}: {rc['size']} users - {rc['reason']}")
            if len(rejected_clusters) > 5:
                print(f"     ... and {len(rejected_clusters)-5} more")
        
        self.roles = roles
        print(f"\n  ✓ Extracted {len(roles)} valid roles")
        
        return roles
    
    def evaluate(self):
        """
        Comprehensive evaluation with advanced metrics
        """
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.user_embeddings
        labels = self.labels
        
        # Filter noise
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]
        
        metrics = {}
        
        # Basic clustering metrics
        if len(set(labels_clean)) > 1 and len(X_clean) > 1:
            metrics['silhouette_score'] = float(silhouette_score(X_clean, labels_clean))
            metrics['davies_bouldin_index'] = float(davies_bouldin_score(X_clean, labels_clean))
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X_clean, labels_clean))
        else:
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_index'] = float('inf')
            metrics['calinski_harabasz_score'] = 0.0
        
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
            metrics['max_cohesion'] = float(np.max(cohesions))
            
            coverages = [r['coverage'] for r in self.roles.values()]
            metrics['avg_coverage_per_role'] = float(np.mean(coverages))
            metrics['min_coverage'] = float(np.min(coverages))
            metrics['max_coverage'] = float(np.max(coverages))
            
            # System-wide coverage
            total_original_perms = sum(r['original_permissions_count'] for r in self.roles.values())
            total_role_perms = sum(r['permission_count'] for r in self.roles.values())
            
            if total_original_perms > 0:
                metrics['system_permission_coverage'] = float(total_role_perms / total_original_perms)
            else:
                metrics['system_permission_coverage'] = 0.0
            
            # Permission compression
            total_unique_original_perms = len(self.data.columns)
            unique_role_perms = set()
            for role in self.roles.values():
                unique_role_perms.update(role['permissions'])
            
            metrics['unique_permissions_in_roles'] = len(unique_role_perms)
            metrics['permission_compression_ratio'] = float(1 - (len(unique_role_perms) / total_unique_original_perms))
            
            # Role overlap analysis
            metrics['avg_role_overlap'] = self._calculate_role_overlap()
        
        # Thresholds applied
        metrics['thresholds'] = {
            'similarity': self.similarity_threshold,
            'grouping': self.grouping_threshold,
            'impact': self.impact_threshold
        }
        
        # GNN-specific metrics
        if self.user_embeddings is not None:
            metrics['embedding_dimension'] = self.user_embeddings.shape[1]
            metrics['embedding_method'] = self.embedding_method
            
            if self.embedding_variance is not None:
                metrics['embedding_variance_explained'] = float(self.embedding_variance)
        
        if self.graph is not None:
            metrics['graph_nodes'] = self.graph.number_of_nodes()
            metrics['graph_edges'] = self.graph.number_of_edges()
        
        self.metrics = metrics
        return metrics
    
    def _calculate_role_overlap(self):
        """
        Calculate average permission overlap between roles
        """
        if len(self.roles) < 2:
            return 0.0
        
        role_names = list(self.roles.keys())
        overlaps = []
        
        for i, role1 in enumerate(role_names):
            for role2 in role_names[i+1:]:
                perms1 = set(self.roles[role1]['permissions'])
                perms2 = set(self.roles[role2]['permissions'])
                
                if len(perms1) == 0 or len(perms2) == 0:
                    continue
                
                intersection = len(perms1 & perms2)
                union = len(perms1 | perms2)
                
                if union > 0:
                    overlap = intersection / union
                    overlaps.append(overlap)
        
        return float(np.mean(overlaps)) if overlaps else 0.0
    
    def visualize_embeddings(self, method='umap', save_path=None):
        """
        Advanced visualization with UMAP or t-SNE
        
        Parameters:
        -----------
        method : str
            'umap', 'tsne', or 'pca'
        """
        if self.user_embeddings is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        print(f"\n Visualizing embeddings with {method.upper()}...")
        
        from sklearn.decomposition import PCA
        
        X = self.user_embeddings
        
        # Dimensionality reduction
        if method == 'umap' and HAS_UMAP:
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            X_2d = reducer.fit_transform(X)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
            X_2d = reducer.fit_transform(X)
        else:
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(X)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Colored by role
        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        for idx, label in enumerate(unique_labels):
            if label == -1:
                color = 'black'
                marker = 'x'
                alpha = 0.3
                size = 30
                label_name = 'Noise'
            else:
                color = colors[idx % len(colors)]
                marker = 'o'
                alpha = 0.7
                size = 50
                label_name = f'Role {label}'
            
            mask = self.labels == label
            ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[color], label=label_name, marker=marker,
                       s=size, alpha=alpha, edgecolors='black', linewidth=0.3)
        
        ax1.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax1.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax1.set_title(f'User Embeddings by Role', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.2)
        
        # Plot 2: Colored by cohesion (for valid roles only)
        valid_mask = self.labels != -1
        cohesion_scores = np.zeros(len(self.labels))
        
        for role_name, role_info in self.roles.items():
            for user in role_info['users']:
                user_idx = self.data.index.get_loc(user)
                cohesion_scores[user_idx] = role_info['cohesion']
        
        scatter = ax2.scatter(X_2d[valid_mask, 0], X_2d[valid_mask, 1],
                            c=cohesion_scores[valid_mask], cmap='RdYlGn',
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        # Noise points
        noise_mask = self.labels == -1
        if noise_mask.sum() > 0:
            ax2.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                       c='black', marker='x', s=30, alpha=0.3, label='Noise')
        
        ax2.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax2.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax2.set_title(f'Role Cohesion Heatmap', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Role Cohesion', fontsize=10)
        ax2.grid(True, alpha=0.2)
        
        plt.suptitle(f'Optimized GNN Embeddings Visualization ({method.upper()})',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  ✓ Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_summary(self):
        """
        Comprehensive summary with quality insights
        """
        if self.roles is None or self.metrics is None:
            print("❌ No results to display")
            return
        
        print("\n" + "="*80)
        print(f" OPTIMIZED GNN ROLE MINER - RESULTS")
        print("="*80)
        
        print(f"\n  Applied Thresholds:")
        print(f"   • Similarity: {self.similarity_threshold}")
        print(f"   • Grouping: {self.grouping_threshold} users")
        print(f"   • Impact: {self.impact_threshold}")
        
        print(f"\n Embedding Configuration:")
        print(f"   • Method: {self.embedding_method}")
        print(f"   • Dimension: {self.metrics.get('embedding_dimension', 'N/A')}")
        if 'embedding_variance_explained' in self.metrics:
            print(f"   • Variance explained: {self.metrics['embedding_variance_explained']:.2%}")
        
        print(f"\n  Clustering Configuration:")
        print(f"   • Method: {self.clustering_method}")
        print(f"   • Refinement iterations: {self.refinement_iterations}")
        print(f"   • Min cluster quality: {self.min_cluster_quality}")
        
        print(f"\n Clustering Quality Metrics:")
        print(f"   • Silhouette score: {self.metrics['silhouette_score']:.3f}")
        print(f"   • Davies-Bouldin index: {self.metrics['davies_bouldin_index']:.3f}")
        print(f"   • Calinski-Harabasz score: {self.metrics.get('calinski_harabasz_score', 0):.1f}")
        
        print(f"\n Role Statistics:")
        print(f"   • Valid roles: {self.metrics['num_roles']}")
        print(f"   • Noise users: {self.metrics['num_noise']} ({self.metrics['noise_percentage']:.1f}%)")
        print(f"   • User coverage: {self.metrics['user_coverage']:.1f}%")
        print(f"   • Avg role size: {self.metrics.get('avg_role_size', 0):.1f} users")
        print(f"   • Role size range: {self.metrics.get('min_role_size', 0)}-{self.metrics.get('max_role_size', 0)}")
        print(f"   • Avg permissions/role: {self.metrics.get('avg_permissions_per_role', 0):.1f}")
        print(f"   • Permission range: {self.metrics.get('min_permissions_per_role', 0)}-{self.metrics.get('max_permissions_per_role', 0)}")
        
        print(f"\n Role Quality Metrics:")
        print(f"   • Avg cohesion: {self.metrics.get('avg_cohesion', 0):.2%}")
        print(f"   • Cohesion range: {self.metrics.get('min_cohesion', 0):.2%}-{self.metrics.get('max_cohesion', 0):.2%}")
        print(f"   • Avg coverage per role: {self.metrics.get('avg_coverage_per_role', 0):.2%}")
        print(f"   • Avg role overlap: {self.metrics.get('avg_role_overlap', 0):.2%}")
        
        print(f"\n  Permission Optimization:")
        print(f"   • System permission coverage: {self.metrics.get('system_permission_coverage', 0):.2%}")
        print(f"   • Unique permissions in roles: {self.metrics.get('unique_permissions_in_roles', 0)}")
        print(f"   • Permission compression: {self.metrics.get('permission_compression_ratio', 0):.2%}")
        
        print(f"\n Detailed Role Breakdown:")
        
        roles_data = []
        for role_name, role_info in sorted(self.roles.items(),
                                          key=lambda x: x[1]['size'],
                                          reverse=True):
            roles_data.append({
                'Role': role_name,
                'Users': role_info['size'],
                'Perms': role_info['permission_count'],
                'Cohesion': f"{role_info['cohesion']:.1%}",
                'Coverage': f"{role_info['coverage']:.1%}",
                'Top_Permissions': ', '.join(role_info['permissions'][:3]) +
                                  (f' (+{role_info["permission_count"]-3})' if role_info['permission_count'] > 3 else '')
            })
        
        df_roles = pd.DataFrame(roles_data)
        print(df_roles.to_string(index=False))


# Example usage
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║            OPTIMIZED GNN ROLE MINER - MAXIMUM QUALITY                    ║
    ║                                                                          ║
    ║  Focus: Best possible results, quality over speed                       ║
    ║  Features:                                                               ║
    ║    • Ensemble embeddings (SVD + Node2Vec + NMF)                         ║
    ║    • Advanced clustering with multiple algorithms                       ║
    ║    • Iterative refinement and quality validation                        ║
    ║    • Automatic parameter tuning with multiple metrics                   ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Example: Maximum quality configuration
    miner = OptimizedGNNRoleMiner(
        # Embedding config
        embedding_dim=128,
        embedding_method='ensemble',  # Best quality
        
        # Graph walks (for Node2Vec/DeepWalk)
        walk_length=80,
        num_walks=100,
        p=1.0,
        q=1.0,
        window_size=10,
        epochs=10,
        
        # Clustering config
        clustering_method='spectral',  # Best for quality
        refinement_iterations=3,
        
        # Quality thresholds
        min_cluster_quality=0.6,
        merge_similar_roles=True,
        role_similarity_threshold=0.85,
        
        # Role mining thresholds
        similarity_threshold=0.7,
        grouping_threshold=5,
        impact_threshold=0.8
    )
    
    # Load data
    miner.load_data('user_permissions.csv')
    
    # Fit with full optimization
    miner.fit(auto_tune=True, verbose=True)
    
    # Display results
    miner.print_summary()
    
    # Advanced visualizations
    miner.visualize_embeddings_advanced(method='umap', save_path='gnn_embeddings_umap.png')
    
    # Export results
    miner.export_results('results/gnn_optimized')
    
    print("\n Optimization complete!")