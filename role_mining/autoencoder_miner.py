# autoencoder_miner.py

import os
# Disable GPU for PoC to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Configure TensorFlow for better CPU performance
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from base_miner import BaseRoleMiner

class AutoencoderRoleMiner(BaseRoleMiner):
    """
    Autoencoder + Clustering for Role Mining
    
    Key Parameters:
    ---------------
    encoding_dim : int
        Dimension of the latent space (bottleneck)
    hidden_layers : list
        Architecture of hidden layers [layer1_size, layer2_size, ...]
    dropout_rate : float
        Dropout rate for regularization
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    """
    
    def __init__(self, encoding_dim=32, hidden_layers=[64, 32], dropout_rate=0.2,
                 epochs=100, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.history = None
        self.latent_representations = None
        self.reconstruction_quality = None
        
    def _build_autoencoder(self, input_dim):
        """Build autoencoder model"""
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        x = input_layer
        for units in self.hidden_layers:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Bottleneck
        encoded = Dense(self.encoding_dim, activation='relu', name='bottleneck')(x)
        
        # Decoder (reverse of encoder)
        x = encoded
        for units in reversed(self.hidden_layers):
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        decoded = Dense(input_dim, activation='sigmoid')(x)
        
        # Models
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        
        # Compile autoencoder
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return autoencoder, encoder
    
    def tune_parameters(self, max_clusters=15, plot=True, save_plot=None):
        """
        Tune number of clusters in latent space
        """
        print("  Tuning Autoencoder parameters...")
        
        if self.latent_representations is None:
            raise ValueError("Autoencoder not trained. Call fit() first.")
        
        X_latent = self.latent_representations
        
        # Test different numbers of clusters
        clusters_range = range(2, min(max_clusters + 1, len(self.data) - 1))
        silhouette_scores = []
        inertia_values = []
        
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in clusters_range:
            try:
                # Apply K-means in latent space
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_latent)
                
                # Calculate silhouette score
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(X_latent, labels)
                    silhouette_scores.append(silhouette)
                    
                    # Update best
                    if silhouette > best_score:
                        best_score = silhouette
                        best_n_clusters = n_clusters
                else:
                    silhouette_scores.append(0)
                
                inertia_values.append(kmeans.inertia_)
                
            except Exception as e:
                print(f"    Warning: Clustering failed for {n_clusters} clusters: {e}")
                silhouette_scores.append(0)
                inertia_values.append(float('inf'))
        
        print(f"    Optimal number of clusters: {best_n_clusters} (silhouette: {best_score:.3f})")
        
        # Plot tuning results
        if plot and len(clusters_range) > 1:
            self._plot_tuning_results(clusters_range, silhouette_scores, 
                                    inertia_values, best_n_clusters, save_plot)
        
        return best_n_clusters
    
    def _plot_tuning_results(self, clusters_range, silhouette_scores, 
                           inertia_values, best_n, save_plot):
        """Plot autoencoder tuning results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Silhouette score
        ax1.plot(clusters_range, silhouette_scores, 'bo-', linewidth=2, markersize=6)
        ax1.axvline(x=best_n, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_n}')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Autoencoder: Silhouette Score vs Number of Clusters')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Elbow curve (inertia)
        ax2.plot(clusters_range, inertia_values, 'go-', linewidth=2, markersize=6)
        ax2.axvline(x=best_n, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_n}')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Inertia')
        ax2.set_title('Autoencoder: Elbow Curve (Inertia)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(save_plot, dpi=150, bbox_inches='tight')
            print(f"    Tuning plot saved: {save_plot}")
        else:
            plt.show()
        
        plt.close()
    
    def fit(self, n_clusters=None, **params):
        """
        Fit autoencoder and cluster in latent space
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters in latent space
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        X = self.data.values.astype(np.float32)
        
        print("  Training Autoencoder...")
        
        # Build and train autoencoder
        input_dim = X.shape[1]
        self.autoencoder, self.encoder = self._build_autoencoder(input_dim)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train autoencoder
        self.history = self.autoencoder.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get latent representations
        self.latent_representations = self.encoder.predict(X, verbose=0)
        
        # Calculate reconstruction quality
        self.reconstruction_quality = self._calculate_reconstruction_quality(X)
        
        print(f"    Autoencoder training completed")
        print(f"    Final loss: {self.history.history['loss'][-1]:.4f}")
        print(f"    Latent space dimension: {self.encoding_dim}")
        print(f"    Reconstruction accuracy: {self.reconstruction_quality['binary_accuracy']:.2%}")
        
        # Cluster in latent space
        if n_clusters is None:
            # Auto-tune number of clusters
            n_clusters = self.tune_parameters(plot=False)
        
        print(f"  Clustering in latent space with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(self.latent_representations)
        
        # Extract roles - THIS IS CRITICAL FOR METRICS
        self.extract_roles()
        
        # Force evaluation to ensure metrics are calculated
        self.evaluate()
        
        print(f"    Clustering complete")
        
        return self.labels
    
    def _calculate_reconstruction_quality(self, X):
        """Calculate reconstruction quality metrics"""
        X_reconstructed = self.autoencoder.predict(X, verbose=0)
        
        # Reconstruction error (MSE)
        mse = np.mean((X - X_reconstructed) ** 2)
        
        # Binary accuracy (since data is binary)
        threshold = 0.5
        binary_predictions = (X_reconstructed > threshold).astype(int)
        binary_accuracy = np.mean(binary_predictions == X)
        
        return {
            'reconstruction_mse': float(mse),
            'binary_accuracy': float(binary_accuracy),
            'final_training_loss': float(self.history.history['loss'][-1]),
            'final_validation_loss': float(self.history.history['val_loss'][-1])
        }
    
    def get_reconstruction_quality(self):
        """Get reconstruction quality metrics"""
        if self.reconstruction_quality is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.reconstruction_quality
    
    def evaluate(self):
        """Override evaluate to include autoencoder-specific metrics"""
        # First call parent evaluation
        super().evaluate()
        
        # Then add autoencoder-specific metrics
        if self.reconstruction_quality is not None and self.metrics is not None:
            self.metrics.update({
                'autoencoder_reconstruction_mse': self.reconstruction_quality['reconstruction_mse'],
                'autoencoder_binary_accuracy': self.reconstruction_quality['binary_accuracy'],
                'autoencoder_training_loss': self.reconstruction_quality['final_training_loss'],
                'autoencoder_validation_loss': self.reconstruction_quality['final_validation_loss'],
                'encoding_dimension': self.encoding_dim
            })
        
        return self.metrics
    
    def visualize_training_history(self, save_path=None):
        """Plot autoencoder training history"""
        if self.history is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Autoencoder Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        if 'accuracy' in self.history.history:
            ax2.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
            ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Autoencoder Training Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Training history plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_latent_space(self, save_path=None):
        """Visualize latent space using PCA"""
        from sklearn.decomposition import PCA
        
        if self.latent_representations is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Reduce latent space to 2D for visualization
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_representations)
        
        plt.figure(figsize=(12, 8))
        
        # Plot with cluster colors
        scatter = plt.scatter(
            latent_2d[:, 0], latent_2d[:, 1],
            c=self.labels, cmap='tab10',
            s=50, alpha=0.7, edgecolors='black', linewidth=0.5
        )
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Latent Dimension 1 (PCA)')
        plt.ylabel('Latent Dimension 2 (PCA)')
        plt.title('Autoencoder Latent Space Visualization\n(Colors represent discovered roles)')
        plt.grid(True, alpha=0.3)
        
        # Add variance explained
        variance_ratio = pca.explained_variance_ratio_
        plt.figtext(0.02, 0.02, f'PCA Variance Explained: {variance_ratio[0]:.1%} + {variance_ratio[1]:.1%} = {sum(variance_ratio):.1%}', 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Latent space plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_summary(self):
        """Extended summary with autoencoder-specific information"""
        super().print_summary()
        
        if self.autoencoder is not None and self.metrics is not None:
            print(f"\n Autoencoder Specific Metrics:")
            print(f"   • Reconstruction MSE: {self.metrics.get('autoencoder_reconstruction_mse', 0):.4f}")
            print(f"   • Binary accuracy: {self.metrics.get('autoencoder_binary_accuracy', 0):.2%}")
            print(f"   • Final training loss: {self.metrics.get('autoencoder_training_loss', 0):.4f}")
            print(f"   • Encoding dimension: {self.metrics.get('encoding_dimension', 0)}")