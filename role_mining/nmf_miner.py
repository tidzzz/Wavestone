# nmf_miner.py

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from base_miner import BaseRoleMiner

class NMFRoleMiner(BaseRoleMiner):
    """
    Non-negative Matrix Factorization for Role Mining
    
    Key Parameters:
    ---------------
    n_components : int
        Number of roles/components to extract
    init : str
        Initialization method ('random', 'nndsvd', 'nndsvda', 'nndsvdar')
    beta_loss : str
        Beta divergence to measure loss ('frobenius', 'kullback-leibler')
    max_iter : int
        Maximum number of iterations
    """
    
    def __init__(self, n_components=8, init='nndsvd', beta_loss='frobenius', 
                 max_iter=1000, **kwargs):  # Increased max_iter for convergence
        super().__init__(**kwargs)
        self.n_components = n_components
        self.init = init
        self.beta_loss = beta_loss
        self.max_iter = max_iter
        self.model = None
        self.W = None  # User-role matrix
        self.H = None  # Role-permission matrix
        self.reconstruction_error = None
        
    def tune_parameters(self, max_components=15, plot=True, save_plot=None):
        """
        Find optimal number of components using reconstruction error and silhouette score
        """
        print("  Tuning NMF parameters...")
        X = self.data.values
        
        # Test different numbers of components
        max_possible = min(max_components, len(self.data) - 1, X.shape[1] - 1)
        if max_possible < 2:
            print("    Warning: Not enough data for tuning, using default 8 components")
            return 8
            
        components_range = range(2, max_possible + 1)
        reconstruction_errors = []
        silhouette_scores = []
        
        best_score = -1
        best_n_components = 2
        
        for n_comp in components_range:
            try:
                # Fit NMF with increased iterations for convergence
                nmf = NMF(
                    n_components=n_comp,
                    init=self.init,
                    beta_loss=self.beta_loss,
                    max_iter=1000,  # Increased for better convergence
                    random_state=42
                )
                W = nmf.fit_transform(X)
                H = nmf.components_
                
                # Calculate reconstruction error
                reconstruction_error = nmf.reconstruction_err_
                reconstruction_errors.append(reconstruction_error)
                
                # Get cluster assignments (assign to role with highest weight)
                labels = np.argmax(W, axis=1)
                
                # Calculate silhouette score
                if len(set(labels)) > 1 and len(X) > 1:
                    try:
                        silhouette = silhouette_score(X, labels)
                        silhouette_scores.append(silhouette)
                        
                        # Update best
                        if silhouette > best_score:
                            best_score = silhouette
                            best_n_components = n_comp
                    except Exception as e:
                        print(f"    Warning: Silhouette failed for {n_comp} components: {e}")
                        silhouette_scores.append(0)
                else:
                    silhouette_scores.append(0)
                    
            except Exception as e:
                print(f"    Warning: Failed for n_components={n_comp}: {e}")
                reconstruction_errors.append(np.nan)
                silhouette_scores.append(0)
        
        # Set optimal number of components
        if best_score > 0:  # Only update if we found a valid solution
            self.n_components = best_n_components
            print(f"    Optimal number of roles: {self.n_components} (silhouette: {best_score:.3f})")
        else:
            print(f"    Using default number of roles: {self.n_components} (no valid solution found)")
        
        # Plot tuning results only if we have valid data
        if plot and len(components_range) > 1:
            self._plot_tuning_results(components_range, reconstruction_errors, 
                                    silhouette_scores, best_n_components, save_plot)
        
        return best_n_components
    
    def _plot_tuning_results(self, components_range, reconstruction_errors, 
                           silhouette_scores, best_n, save_plot):
        """Plot NMF tuning results"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Reconstruction error
            ax1.plot(components_range, reconstruction_errors, 'bo-', linewidth=2, markersize=6)
            ax1.axvline(x=best_n, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_n}')
            ax1.set_xlabel('Number of Components')
            ax1.set_ylabel('Reconstruction Error')
            ax1.set_title('NMF: Reconstruction Error vs Number of Components')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Silhouette score
            ax2.plot(components_range, silhouette_scores, 'go-', linewidth=2, markersize=6)
            ax2.axvline(x=best_n, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_n}')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('NMF: Silhouette Score vs Number of Components')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(save_plot, dpi=150, bbox_inches='tight')
                print(f"    Tuning plot saved: {save_plot}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not create tuning plot: {e}")
    
    def fit(self, n_components=None, **params):
        """
        Fit NMF model and extract roles
        
        Parameters:
        -----------
        n_components : int
            Number of roles/components (uses tuned value if None)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        X = self.data.values
        
        # Use provided n_components or tuned value
        if n_components is not None:
            self.n_components = n_components
        
        print(f"  Fitting NMF with {self.n_components} components...")
        
        try:
            # Fit NMF with error handling
            self.model = NMF(
                n_components=self.n_components,
                init=self.init,
                beta_loss=self.beta_loss,
                max_iter=self.max_iter,
                random_state=42
            )
            
            self.W = self.model.fit_transform(X)  # User-role matrix
            self.H = self.model.components_       # Role-permission matrix
            self.reconstruction_error = self.model.reconstruction_err_
            
            # Assign users to roles based on highest weight
            self.labels = np.argmax(self.W, axis=1)
            
            # Validate labels were created
            if self.labels is None:
                raise ValueError("NMF failed to create labels")
                
            print(f"    Reconstruction error: {self.reconstruction_error:.4f}")
            print(f"    Role assignments complete")
            
            # Extract roles
            roles = self.extract_roles()
            
            # Force evaluation to ensure metrics are calculated
            self.evaluate()
            
            return self.labels
            
        except Exception as e:
            print(f"     NMF fitting failed: {e}")
            # Set default labels to avoid None issues
            self.labels = np.zeros(len(X), dtype=int)
            self.roles = {}
            self.metrics = {}
            raise e
    
    def extract_roles(self):
        """
        Override extract_roles to handle NMF-specific cases
        """
        try:
            return super().extract_roles()
        except Exception as e:
            print(f"     Role extraction failed: {e}")
            # Return empty roles but don't break the flow
            self.roles = {}
            return {}
    
    def evaluate(self):
        """
        Override evaluate to ensure metrics are always set
        """
        try:
            metrics = super().evaluate()
            
            # Add NMF-specific metrics if available
            if self.reconstruction_error is not None:
                metrics['nmf_reconstruction_error'] = float(self.reconstruction_error)
                metrics['nmf_components'] = self.n_components
                
            return metrics
            
        except Exception as e:
            print(f"     Evaluation failed: {e}")
            # Return basic metrics to avoid complete failure
            basic_metrics = {
                'silhouette_score': 0.0,
                'davies_bouldin_index': float('inf'),
                'num_roles': len(self.roles) if hasattr(self, 'roles') and self.roles else 0,
                'num_noise': 0,
                'noise_percentage': 0.0,
                'user_coverage': 0.0
            }
            self.metrics = basic_metrics
            return basic_metrics
    
    def get_role_definitions(self):
        """
        Get role definitions from H matrix (role-permission matrix)
        """
        if self.H is None:
            print("    Warning: H matrix not available, returning empty definitions")
            return {}
        
        role_definitions = {}
        
        try:
            for role_idx in range(self.H.shape[0]):
                # Get top permissions for this role (based on weights in H matrix)
                role_weights = self.H[role_idx]
                top_permissions_idx = np.argsort(role_weights)[::-1]  # Sort descending
                
                # Get permission names and weights
                permissions = []
                weights = []
                
                for perm_idx in top_permissions_idx:
                    if role_weights[perm_idx] > 0:  # Only include positive weights
                        perm_name = self.data.columns[perm_idx]
                        permissions.append(perm_name)
                        weights.append(role_weights[perm_idx])
                
                role_definitions[f"Role_{role_idx}"] = {
                    'permissions': permissions,
                    'weights': weights,
                    'top_permissions': permissions[:10],  # Top 10 permissions
                    'total_weight': sum(weights)
                }
        except Exception as e:
            print(f"    Warning: Error creating role definitions: {e}")
            
        return role_definitions
    
    def visualize_role_components(self, top_n=10, save_path=None):
        """
        Visualize role components (top permissions per role)
        """
        if self.H is None:
            print("    Cannot visualize: H matrix not available")
            return
        
        role_defs = self.get_role_definitions()
        if not role_defs:
            print("    No role definitions to visualize")
            return
        
        try:
            n_roles = len(role_defs)
            n_cols = min(3, n_roles)
            n_rows = (n_roles + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_roles > 1:
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten()
            else:
                axes = [axes]
            
            for i, (role_name, role_info) in enumerate(role_defs.items()):
                if i >= len(axes):
                    break
                    
                top_perms = role_info['top_permissions'][:top_n]
                top_weights = role_info['weights'][:top_n]
                
                if not top_perms:
                    continue
                    
                # Create horizontal bar plot
                y_pos = np.arange(len(top_perms))
                axes[i].barh(y_pos, top_weights, align='center', alpha=0.7)
                axes[i].set_yticks(y_pos)
                axes[i].set_yticklabels(top_perms, fontsize=8)
                axes[i].invert_yaxis()
                axes[i].set_xlabel('Weight')
                axes[i].set_title(f'{role_name}\nTop {min(top_n, len(top_perms))} Permissions')
                axes[i].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(role_defs), len(axes)):
                if hasattr(axes[i], 'set_visible'):
                    axes[i].set_visible(False)
            
            plt.suptitle(f'NMF Role Components - Top {top_n} Permissions per Role', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f" Role components plot saved: {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"    Warning: Visualization failed: {e}")
    
    def print_summary(self):
        """Extended summary with NMF-specific information"""
        try:
            super().print_summary()
        except Exception as e:
            print(f"     Base summary failed: {e}")
            print("  Falling back to basic NMF summary...")
            print(f"\n NMF Basic Results:")
            print(f"   • Components (roles): {self.n_components}")
            print(f"   • Valid roles: {len(self.roles) if hasattr(self, 'roles') and self.roles else 0}")
        
        if hasattr(self, 'H') and self.H is not None:
            print(f"\n NMF Specific Metrics:")
            print(f"   • Reconstruction error: {self.reconstruction_error:.4f}")
            print(f"   • Components (roles): {self.n_components}")
            
            # Show top permissions for each role
            try:
                role_defs = self.get_role_definitions()
                if role_defs:
                    print(f"\n Top Permissions per Role:")
                    for role_name, role_info in list(role_defs.items())[:5]:  # Show first 5 only
                        top_perms = ", ".join(role_info['top_permissions'][:3])
                        print(f"   • {role_name}: {top_perms}...")
            except Exception as e:
                print(f"   • Could not display role details: {e}")