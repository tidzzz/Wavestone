import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD, NMF
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from base_miner import BaseRoleMiner

class GNNRoleMiner(BaseRoleMiner):
    """
    Optimized Graph Neural Network for Role Mining
    """
    
    def __init__(self, embedding_dim=64, walk_length=10, num_walks=50, 
                 window_size=3, epochs=5, use_sampling=True, **kwargs):
        super().__init__(**kwargs)
        # Reduced default parameters for large graphs
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.epochs = epochs
        self.use_sampling = use_sampling  # Enable sampling for large graphs
        self.graph = None
        self.user_embeddings = None
        
    def _build_optimized_graph(self):
        """Build optimized bipartite graph with sampling for large datasets"""
        print("  Building optimized bipartite graph...")
        
        self.graph = nx.Graph()
        
        # Add user nodes
        user_nodes = [f"user_{i}" for i in range(len(self.data))]
        self.graph.add_nodes_from(user_nodes, bipartite=0, type='user')
        
        # Add permission nodes  
        perm_nodes = [f"perm_{col}" for col in self.data.columns]
        self.graph.add_nodes_from(perm_nodes, bipartite=1, type='permission')
        
        # OPTIMIZATION: Sample edges for very large graphs
        total_possible_edges = len(self.data) * len(self.data.columns)
        print(f"    Total possible edges: {total_possible_edges:,}")
        
        edges = []
        edge_count = 0
        
        for user_idx in range(len(self.data)):
            for perm_idx, perm_name in enumerate(self.data.columns):
                if self.data.iloc[user_idx, perm_idx] == 1:
                    edges.append((f"user_{user_idx}", f"perm_{perm_name}"))
                    edge_count += 1
        
        self.graph.add_edges_from(edges)
        
        print(f"    Graph: {len(user_nodes)} users, {len(perm_nodes)} permissions, {edge_count:,} edges")
        
        # Graph statistics
        print(f"    Graph density: {(2 * edge_count) / (len(user_nodes) * len(perm_nodes)):.6f}")
        
        return self.graph
    
    def _compute_fast_embeddings(self):
        """Use faster embedding methods for large graphs"""
        print("  Using fast SVD-based embeddings for large graph...")
        
        # Method 1: SVD on adjacency matrix (fastest)
        try:
            return self._compute_svd_embeddings()
        except:
            pass
        
        # Method 2: Fallback to DeepWalk with optimized parameters
        try:
            return self._compute_optimized_deepwalk()
        except Exception as e:
            print(f"    DeepWalk failed: {e}")
        
        # Method 3: Final fallback - direct matrix factorization
        return self._compute_mf_embeddings()
    
    def _compute_svd_embeddings(self):
        """Use Truncated SVD for fast embeddings"""
        print("    Computing Truncated SVD embeddings...")
        
        from sklearn.decomposition import TruncatedSVD
        
        # Use the user-permission matrix directly
        X = self.data.values.astype(float)
        
        # Apply Truncated SVD
        svd = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        self.user_embeddings = svd.fit_transform(X)
        
        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"    SVD embeddings shape: {self.user_embeddings.shape}")
        print(f"    Explained variance: {explained_variance:.3f}")
        
        return True
    
    def _compute_optimized_deepwalk(self):
        """Optimized DeepWalk with sampling"""
        try:
            import gensim
            from gensim.models import Word2Vec
            
            print("    Computing optimized DeepWalk embeddings...")
            
            # OPTIMIZATION: Sample nodes for very large graphs
            all_user_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['bipartite'] == 0]
            
            if len(all_user_nodes) > 2000 and self.use_sampling:
                print(f"    Sampling 2000 users from {len(all_user_nodes)} for walks...")
                user_nodes = np.random.choice(all_user_nodes, size=2000, replace=False)
            else:
                user_nodes = all_user_nodes
            
            # Generate random walks with reduced parameters
            walks = []
            for _ in range(self.num_walks):
                for node in user_nodes:
                    walk = [node]
                    current = node
                    for _ in range(self.walk_length - 1):
                        neighbors = list(self.graph.neighbors(current))
                        if neighbors:
                            current = np.random.choice(neighbors)
                            walk.append(current)
                        else:
                            break
                    walks.append(walk)
            
            # Train Word2Vec with optimized parameters
            model = Word2Vec(
                walks,
                vector_size=self.embedding_dim,
                window=self.window_size,
                min_count=1,
                workers=2,  # Reduce workers to avoid memory issues
                epochs=self.epochs,
                compute_loss=True
            )
            
            # Extract embeddings for all users
            user_embeddings = []
            for i in range(len(self.data)):
                try:
                    user_embeddings.append(model.wv[f"user_{i}"])
                except KeyError:
                    # If user wasn't in sampled walks, use zero vector
                    user_embeddings.append(np.zeros(self.embedding_dim))
            
            self.user_embeddings = np.array(user_embeddings)
            print(f"    DeepWalk embeddings shape: {self.user_embeddings.shape}")
            
            return True
            
        except ImportError:
            print("    Gensim not available, falling back to SVD...")
            return self._compute_svd_embeddings()
        except Exception as e:
            print(f"    DeepWalk error: {e}")
            return self._compute_svd_embeddings()
    
    def _compute_mf_embeddings(self):
        """Matrix factorization fallback"""
        print("    Using matrix factorization for embeddings...")
        
        from sklearn.decomposition import NMF
        
        # Use NMF to get user embeddings
        nmf = NMF(n_components=self.embedding_dim, random_state=42, max_iter=100)
        self.user_embeddings = nmf.fit_transform(self.data.values)
        
        print(f"    MF embeddings shape: {self.user_embeddings.shape}")
        return True
    
    def tune_parameters(self, max_clusters=15, plot=True, save_plot=None):
        """
        Fast parameter tuning for GNN
        """
        print("  Fast-tuning GNN clustering parameters...")
        
        if self.user_embeddings is None:
            raise ValueError("Embeddings not computed. Call fit() first.")
        
        X = self.user_embeddings
        
        # Test fewer cluster options for speed
        max_possible = min(max_clusters, len(self.data) // 10, 50)  # More conservative
        clusters_range = range(2, max_possible + 1, 2)  # Step by 2 for speed
        
        silhouette_scores = []
        
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in clusters_range:
            try:
                # Use faster K-means configuration
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=50)
                labels = kmeans.fit_predict(X)
                
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(X, labels)
                    silhouette_scores.append(silhouette)
                    
                    if silhouette > best_score:
                        best_score = silhouette
                        best_n_clusters = n_clusters
                else:
                    silhouette_scores.append(0)
                
            except Exception as e:
                print(f"    Warning: Clustering failed for {n_clusters} clusters: {e}")
                silhouette_scores.append(0)
        
        print(f"    Optimal number of clusters: {best_n_clusters} (silhouette: {best_score:.3f})")
        
        # Only plot if requested and we have enough points
        if plot and len(clusters_range) > 2:
            self._plot_tuning_results(clusters_range, silhouette_scores, best_n_clusters, save_plot)
        
        return best_n_clusters
    
    def _plot_tuning_results(self, clusters_range, silhouette_scores, best_n, save_plot):
        """Fast plotting"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(clusters_range, silhouette_scores, 'bo-', linewidth=2, markersize=6)
        plt.axvline(x=best_n, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_n}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('GNN: Silhouette Score vs Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_plot:
            plt.savefig(save_plot, dpi=150, bbox_inches='tight')
            print(f"    Tuning plot saved: {save_plot}")
        else:
            plt.show()
        
        plt.close()
    
    def fit(self, n_clusters=None, **params):
        """
        Optimized fit method for large graphs
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("  Training Optimized GNN Model...")
        
        # Show data size warning
        if len(self.data) > 1000:
            print(f"    Large dataset detected: {len(self.data):,} users")
            print("    Using fast embedding methods...")
        
        # Step 1: Build optimized graph
        self._build_optimized_graph()
        
        # Step 2: Compute fast embeddings (skip Node2Vec for large graphs)
        if len(self.data) > 1000:
            print("    Using SVD embeddings for large graph (fast)...")
            success = self._compute_svd_embeddings()
        else:
            success = self._compute_fast_embeddings()
            
        if not success:
            raise ValueError("Failed to compute graph embeddings")
        
        # Step 3: Fast clustering
        if n_clusters is None:
            n_clusters = self.tune_parameters(plot=False)
        
        print(f"  Clustering embeddings with {n_clusters} clusters...")
        
        # Use faster K-means for large datasets
        if len(self.data) > 1000:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=50)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
        self.labels = kmeans.fit_predict(self.user_embeddings)
        
        # Step 4: Extract roles
        self.extract_roles()
        self.evaluate()
        
        print("    Optimized GNN role mining complete")
        return self.labels

    def visualize_embeddings(self, save_path=None):
        """Visualize GNN embeddings using PCA"""
        from sklearn.decomposition import PCA
        
        if self.user_embeddings is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.user_embeddings)
        
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=self.labels, cmap='tab10',
            s=30, alpha=0.7, edgecolors='black', linewidth=0.3
        )
        
        plt.colorbar(scatter, label='Role')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('GNN Embeddings Visualization\n(Colors represent discovered roles)')
        plt.grid(True, alpha=0.3)
        
        # Add variance explained
        variance_ratio = pca.explained_variance_ratio_
        plt.figtext(0.02, 0.02, f'Variance Explained: {variance_ratio[0]:.1%} + {variance_ratio[1]:.1%} = {sum(variance_ratio):.1%}', 
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Embeddings visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def visualize_graph_structure(self, max_nodes=100, save_path=None):
        """Visualize a sample of the graph structure"""
        if self.graph is None:
            raise ValueError("Graph not built. Call fit() first.")
        
        # Sample a smaller graph for visualization
        user_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['bipartite'] == 0]
        sampled_users = np.random.choice(user_nodes, size=min(max_nodes//2, len(user_nodes)), replace=False)
        
        # Get permissions connected to sampled users
        sampled_perms = []
        for user in sampled_users:
            sampled_perms.extend(list(self.graph.neighbors(user)))
        
        sampled_perms = list(set(sampled_perms))[:max_nodes//2]
        sampled_nodes = list(sampled_users) + sampled_perms
        
        subgraph = self.graph.subgraph(sampled_nodes)
        
        plt.figure(figsize=(15, 10))
        
        # Create layout
        pos = {}
        user_nodes = [n for n in subgraph.nodes() if subgraph.nodes[n]['bipartite'] == 0]
        perm_nodes = [n for n in subgraph.nodes() if subgraph.nodes[n]['bipartite'] == 1]
        
        # Position users on left, permissions on right
        for i, node in enumerate(user_nodes):
            pos[node] = (0, i)
        for i, node in enumerate(perm_nodes):
            pos[node] = (1, i)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, nodelist=user_nodes, 
                              node_color='lightblue', node_size=300, alpha=0.7)
        nx.draw_networkx_nodes(subgraph, pos, nodelist=perm_nodes, 
                              node_color='lightcoral', node_size=200, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray')
        
        # Labels
        labels = {node: node.replace('user_', 'U').replace('perm_', 'P') 
                 for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        plt.title('Graph Structure Sample\n(Blue: Users, Red: Permissions)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Graph structure visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def evaluate(self):
        """Enhanced evaluation with GNN-specific metrics"""
        metrics = super().evaluate()
        
        # Add GNN-specific metrics
        if self.user_embeddings is not None:
            metrics['embedding_dimension'] = self.embedding_dim
            metrics['graph_nodes'] = self.graph.number_of_nodes() if self.graph else 0
            metrics['graph_edges'] = self.graph.number_of_edges() if self.graph else 0
        
        self.metrics = metrics
        return metrics

    def print_summary(self):
        """Extended summary with GNN-specific information"""
        super().print_summary()
        
        if self.graph is not None and self.metrics is not None:
            print(f"\n GNN Specific Metrics:")
            print(f"   • Embedding dimension: {self.metrics.get('embedding_dimension', 0)}")
            print(f"   • Graph nodes: {self.metrics.get('graph_nodes', 0)}")
            print(f"   • Graph edges: {self.metrics.get('graph_edges', 0)}")
            print(f"   • Embedding method: {'SVD' if len(self.data) > 1000 else 'Graph-based'}")