# main.py

import os
# Disable GPU to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import sys
from pathlib import Path

from dbscan_miner import DBSCANRoleMiner
from kmeans_miner import KMeansRoleMiner
from nmf_miner import NMFRoleMiner
from autoencoder_miner import AutoencoderRoleMiner
from gnn_miner import GNNRoleMiner

def main():
    parser = argparse.ArgumentParser(
        description=' AI Role Mining CLI - Intelligent Access Control Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Quick Start Examples:
  
  # Interactive mode (recommended for PoC)
  python main.py --data user_permissions.csv --interactive
  
  # GNN with custom thresholds
  python main.py --data out/user_permission_matrix.csv --algorithm gnn \\
      --similarity 0.7 --grouping 5 --impact 0.8
  
  # Autoencoder with specific parameters
  python main.py --data out/user_permission_matrix.csv --algorithm autoencoder \\
      --epochs 100 --encoding-dim 32
  
  # Compare all AI models
  python main.py --data out/user_permission_matrix.csv --compare

 Threshold Guide:
  --similarity (0.6-0.8) : How similar users must be (higher = stricter)
  --grouping   (3-10)    : Minimum users per role  
  --impact     (0.7-0.9) : Permission inclusion threshold

 AI Models:
  • kmeans       - Fast, interpretable, spherical clusters
  • dbscan       - Auto-detects roles, handles outliers
  • nmf          - Interpretable matrix factorization
  • autoencoder  - Deep learning, complex patterns
  • gnn          - Graph neural networks, relationship learning
        """
    )
    
    # Required arguments
    parser.add_argument('--data', required=True,
                       help='Path to user-permission matrix (CSV/XLSX)')
    
    # Algorithm selection
    parser.add_argument('--algorithm', 
                       choices=['kmeans', 'dbscan', 'nmf', 'autoencoder', 'gnn'],
                       help='AI model for role mining')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare all AI models and recommend best')
    
    parser.add_argument('--interactive', action='store_true',
                       help=' Interactive PoC mode with guided setup')
    
    # Core threshold parameters (the important ones!)
    parser.add_argument('--similarity', type=float, default=0.7,
                       help='User similarity threshold (0-1, default: 0.7)')
    
    parser.add_argument('--grouping', type=int, default=5,
                       help='Minimum users per role (default: 5)')
    
    parser.add_argument('--impact', type=float, default=0.8,
                       help='Permission inclusion threshold (0-1, default: 0.8)')
    
    # Algorithm-specific parameters
    parser.add_argument('--n-clusters', type=int,
                       help='Number of roles (K-means/Hierarchical)')
    
    parser.add_argument('--n-components', type=int,
                       help='Number of components (NMF)')
    
    parser.add_argument('--encoding-dim', type=int, default=32,
                       help='Latent dimension for autoencoder (default: 32)')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs for autoencoder (default: 100)')
    
    # GNN-specific parameters
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension for GNN (default: 64)')
    
    parser.add_argument('--walk-length', type=int, default=10,
                       help='Walk length for Node2Vec (default: 10)')
    
    parser.add_argument('--num-walks', type=int, default=100,
                       help='Number of walks per node (default: 100)')
    
    parser.add_argument('--window-size', type=int, default=3,
                       help='Window size for Node2Vec (default: 3)')
    
    parser.add_argument('--metric', default='cosine',
                       choices=['cosine', 'jaccard', 'euclidean'],
                       help='Distance metric for DBSCAN (default: cosine)')
    
    # Visualization & analysis
    parser.add_argument('--plot', action='store_true',
                       help='Generate role visualization plots')
    
    parser.add_argument('--show-components', action='store_true',
                       help='Show role components (NMF only)')
    
    parser.add_argument('--show-latent', action='store_true',
                       help='Show latent space (Autoencoder only)')
    
    parser.add_argument('--show-embeddings', action='store_true',
                       help='Show GNN embeddings visualization')
    
    parser.add_argument('--show-graph', action='store_true',
                       help='Show graph structure (GNN only)')
    
    # Output configuration
    parser.add_argument('--output', default='results',
                       help='Output directory (default: results)')
    
    parser.add_argument('--no-export', action='store_true',
                       help='Skip exporting results (faster iteration)')
    
    args = parser.parse_args()
    
    # Validate data file
    if not Path(args.data).exists():
        print(f" Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        run_interactive_poc(args)
        return
    
    # Compare all models
    if args.compare:
        compare_ai_models(args)
        return
    
    # Single algorithm mode
    if not args.algorithm:
        print(" Error: Specify --algorithm or use --compare/--interactive")
        parser.print_help()
        sys.exit(1)
    
    run_single_algorithm(args)


def run_interactive_poc(args):
    """Interactive Proof-of-Concept mode with guided setup"""
    print("\n" + "="*80)
    print(" AI ROLE MINING - INTERACTIVE PoC MODE")
    print("="*80)
    
    # Step 1: Data overview
    print(f"\n DATA OVERVIEW")
    print(f"   File: {Path(args.data).name}")
    
    # Quick data peek
    try:
        import pandas as pd
        data = pd.read_csv(args.data, index_col=0) if args.data.endswith('.csv') else pd.read_excel(args.data, index_col=0)
        print(f"   Users: {len(data):,} | Permissions: {len(data.columns):,}")
        print(f"   Sparsity: {(1 - data.sum().sum() / data.size) * 100:.1f}%")
    except:
        print("   (Data preview unavailable)")
    
    # Step 2: Use case selection
    print(f"\n SELECT YOUR USE CASE:")
    print("   1. Quick Baseline (Fastest, most interpretable)")
    print("   2. Smart Discovery (Auto-detects roles, handles outliers)") 
    print("   3. Interpretable Roles (Matrix factorization, business-friendly)")
    print("   4. Advanced Patterns (Deep learning, complex relationships)")
    print("   5. Graph AI (Relationship learning, state-of-the-art)")
    print("   6. Compare All Models (Find optimal approach)")
    
    choice = input("\nSelect approach (1-6): ").strip()
    
    use_case_map = {
        '1': ('kmeans', " Quick Baseline - K-means"),
        '2': ('dbscan', " Smart Discovery - DBSCAN"), 
        '3': ('nmf', " Interpretable Roles - NMF"),
        '4': ('autoencoder', " Advanced Patterns - Autoencoder"),
        '5': ('gnn', " Graph AI - GNN"),
        '6': ('compare', " Compare All AI Models - Compare")
    }
    
    if choice not in use_case_map:
        print(" Invalid choice")
        return
    
    algo, description = use_case_map[choice]
    
    if algo == 'compare':
        args.compare = True
    else:
        args.algorithm = algo
    
    print(f"\n Selected: {description}")
    
    # Step 3: Configure for use case
    print(f"\n  CONFIGURATION FOR {description.split(' - ')[1]}")
    
    # Core threshold configuration
    print(f"\n CORE THRESHOLDS (Applied to all algorithms)")
    
    sim_input = input(f"   Similarity threshold (0.5-0.9, Enter for {args.similarity}): ").strip()
    if sim_input:
        args.similarity = float(sim_input)
    
    group_input = input(f"   Grouping threshold (3-10, Enter for {args.grouping}): ").strip()
    if group_input:
        args.grouping = int(group_input)
    
    impact_input = input(f"   Impact threshold (0.6-0.9, Enter for {args.impact}): ").strip()
    if impact_input:
        args.impact = float(impact_input)
    
    # Algorithm-specific configurations
    if algo == 'kmeans':
        print(f"\n K-MEANS SPECIFIC")
        print("   K-means works best with 8-15 roles for most organizations")
        clusters = input("   How many roles to discover? (press Enter for auto-tune): ").strip()
        if clusters:
            args.n_clusters = int(clusters)
    
    elif algo == 'dbscan':
        print(f"\n DBSCAN SPECIFIC")
        print("   DBSCAN auto-detects roles based on density")
        print("   • Metric: How to measure user similarity")
        metric_choice = input(f"   Distance metric (cosine/jaccard/euclidean, Enter for {args.metric}): ").strip()
        if metric_choice:
            args.metric = metric_choice
    
    elif algo == 'nmf': 
        print(f"\n NMF SPECIFIC")
        print("   NMF discovers interpretable role components")
        print("   • Components: Number of roles to extract")
        comp_input = input("   Number of components? (press Enter for auto-tune): ").strip()
        if comp_input:
            args.n_components = int(comp_input)
    
    elif algo == 'autoencoder':
        print(f"\n AUTOENCODER SPECIFIC")
        print("   Autoencoder learns complex permission patterns")
        epochs_input = input(f"   Training epochs (50-200, Enter for {args.epochs}): ").strip()
        if epochs_input:
            args.epochs = int(epochs_input)
        
        encoding_input = input(f"   Encoding dimension (16-64, Enter for {args.encoding_dim}): ").strip()
        if encoding_input:
            args.encoding_dim = int(encoding_input)
    
    elif algo == 'gnn':
        print(f"\n GNN SPECIFIC")
        print("   GNN learns from user-permission relationships")
        embed_input = input(f"   Embedding dimension (64-256, Enter for {args.embedding_dim}): ").strip()
        if embed_input:
            args.embedding_dim = int(embed_input)
        
        walk_input = input(f"   Walk length (20-50, Enter for {args.walk_length}): ").strip()
        if walk_input:
            args.walk_length = int(walk_input)
    
    # Step 4: Output options
    print(f"\n OUTPUT OPTIONS")
    args.plot = input("   Generate visualizations? (y/n, default y): ").strip().lower() != 'n'
    
    # Algorithm-specific visualizations
    if algo == 'nmf':
        args.show_components = input("   Show role components? (y/n, default y): ").strip().lower() != 'n'
    elif algo == 'autoencoder':
        args.show_latent = input("   Show latent space? (y/n, default y): ").strip().lower() != 'n'
    elif algo == 'gnn':
        args.show_embeddings = input("   Show embeddings? (y/n, default y): ").strip().lower() != 'n'
        args.show_graph = input("   Show graph structure? (y/n, default n): ").strip().lower() == 'y'
    
    output_dir = input(f"   Output directory (Enter for '{args.output}'): ").strip()
    if output_dir:
        args.output = output_dir
    
    # Summary
    print(f"\n PoC CONFIGURATION SUMMARY")
    print(f"   • Approach: {description}")
    print(f"   • Data: {Path(args.data).name}")
    print(f"   • Similarity: {args.similarity}")
    print(f"   • Grouping: Min {args.grouping} users per role")
    print(f"   • Impact: {args.impact}")
    print(f"   • Visualizations: {'Yes' if args.plot else 'No'}")
    print(f"   • Output: {args.output}/")
    
    if algo == 'autoencoder':
        print(f"   • Training: {args.epochs} epochs (CPU-only)")
    elif algo == 'gnn':
        print(f"   • Embedding: {args.embedding_dim} dimensions")
    
    input("\nPress Enter to start AI role mining...")
    
    # Run the selected approach
    if args.compare:
        compare_ai_models(args)
    else:
        run_single_algorithm(args)


def run_single_algorithm(args):
    """Run a single AI clustering algorithm"""
    print("\n" + "="*80)
    print(f" RUNNING {args.algorithm.upper()} ROLE MINING")
    print("="*80)
    
    # Initialize the selected AI miner with core thresholds
    miner_params = {
        'similarity_threshold': args.similarity,
        'grouping_threshold': args.grouping,
        'impact_threshold': args.impact
    }
    
    if args.algorithm == 'kmeans':
        miner = KMeansRoleMiner(**miner_params)
        algo_name = " K-means Clustering"
    elif args.algorithm == 'dbscan':
        miner = DBSCANRoleMiner(**miner_params)
        algo_name = " DBSCAN Density Clustering"
    elif args.algorithm == 'nmf':
        miner = NMFRoleMiner(**miner_params)
        algo_name = " Non-negative Matrix Factorization"
    elif args.algorithm == 'autoencoder':
        miner = AutoencoderRoleMiner(
            encoding_dim=args.encoding_dim,
            epochs=args.epochs,
            **miner_params
        )
        algo_name = " Deep Autoencoder (CPU Mode)"
    elif args.algorithm == 'gnn':
        miner = GNNRoleMiner(
            embedding_dim=args.embedding_dim,
            walk_length=args.walk_length,
            num_walks=args.num_walks,
            window_size=args.window_size,
            **miner_params
        )
        algo_name = " Graph Neural Network"
    
    print(f"\n{algo_name}")
    print(f"   Similarity: {args.similarity} | Grouping: {args.grouping} | Impact: {args.impact}")
    
    # Load and analyze data
    print(f"\n LOADING DATA...")
    data = miner.load_data(args.data)
    
    # Algorithm-specific tuning
    print(f"\n  AI MODEL OPTIMIZATION...")
    
    fit_params = {}
    
    if args.algorithm == 'dbscan':
        miner.tune_parameters(
            metric=args.metric,
            plot=args.plot,
            save_plot=f'{args.output}/dbscan_tuning.png' if args.plot else None
        )
        fit_params['metric'] = args.metric
        
    elif args.algorithm == 'kmeans':
        miner.tune_parameters(
            plot=args.plot,
            save_plot=f'{args.output}/kmeans_tuning.png' if args.plot else None
        )
        if args.n_clusters:
            fit_params['n_clusters'] = args.n_clusters
        
    elif args.algorithm == 'nmf':
        miner.tune_parameters(
            plot=args.plot,
            save_plot=f'{args.output}/nmf_tuning.png' if args.plot else None
        )
        if args.n_components:
            fit_params['n_components'] = args.n_components
        
    elif args.algorithm == 'autoencoder':
        # Autoencoder tunes during fit
        if args.n_clusters:
            fit_params['n_clusters'] = args.n_clusters
        print("   Autoencoder will tune during training...")
        
    elif args.algorithm == 'gnn':
        # GNN tunes during fit
        if args.n_clusters:
            fit_params['n_clusters'] = args.n_clusters
        print("   GNN will tune during training...")
    
    # Train the AI model
    print(f"\n TRAINING AI MODEL...")
    
    # Special handling for slower algorithms
    if args.algorithm == 'autoencoder':
        print("    Training on CPU (this may take 2-5 minutes)...")
    elif args.algorithm == 'gnn':
        print("    Building graph and computing embeddings...")
    
    miner.fit(**fit_params)
    
    # Show results
    print(f"\n ROLE MINING RESULTS")
    miner.print_summary()
    
    # Generate visualizations
    if args.plot:
        print(f"\n GENERATING VISUALIZATIONS...")
        
        # Main cluster visualization
        miner.visualize_results(
            save_path=f'{args.output}/{args.algorithm}_clusters.png'
        )
        
        # Algorithm-specific visualizations
        if args.algorithm == 'nmf' and args.show_components:
            miner.visualize_role_components(
                save_path=f'{args.output}/nmf_components.png'
            )
        elif args.algorithm == 'autoencoder':
            if args.show_latent:
                miner.visualize_latent_space(
                    save_path=f'{args.output}/autoencoder_latent.png'
                )
            miner.visualize_training_history(
                save_path=f'{args.output}/autoencoder_training.png'
            )
        elif args.algorithm == 'gnn':
            if args.show_embeddings:
                miner.visualize_embeddings(
                    save_path=f'{args.output}/gnn_embeddings.png'
                )
            if args.show_graph:
                miner.visualize_graph_structure(
                    save_path=f'{args.output}/gnn_graph_structure.png'
                )
    
    # Export results
    if not args.no_export:
        print(f"\n EXPORTING RESULTS...")
        Path(args.output).mkdir(exist_ok=True)
        miner.export_results(args.output)
        
        # Special exports for AI models
        if args.algorithm == 'nmf':
            role_defs = miner.get_role_definitions()
            print(f"    Role components exported")
        elif args.algorithm == 'autoencoder':
            recon_quality = miner.get_reconstruction_quality()
            print(f"    Reconstruction accuracy: {recon_quality['binary_accuracy']:.2%}")
        elif args.algorithm == 'gnn':
            print(f"    Graph embeddings and structure exported")
    
    print(f"\n {algo_name} COMPLETED!")
    print(f"   Results saved to: {args.output}/")


def compare_ai_models(args):
    """Compare all five AI models and recommend best approach"""
    print("\n" + "="*80)
    print(" COMPARING AI ROLE MINING MODELS")
    print("="*80)
    print(" Running all models in CPU-only mode for compatibility")
    
    algorithms = [
        ('kmeans', ' K-means'),
        ('dbscan', ' DBSCAN'), 
        ('nmf', ' NMF'),
        ('autoencoder', ' Autoencoder'),
        ('gnn', ' GNN')
    ]
    
    results = {}
    
    for algo, display_name in algorithms:
        print(f"\n{'─'*80}")
        print(f"Testing: {display_name}")
        print('─'*80)
        
        try:
            # Initialize miner with common parameters
            miner_params = {
                'similarity_threshold': args.similarity,
                'grouping_threshold': args.grouping,
                'impact_threshold': args.impact
            }
            
            if algo == 'kmeans':
                miner = KMeansRoleMiner(**miner_params)
            elif algo == 'dbscan':
                miner = DBSCANRoleMiner(**miner_params)
            elif algo == 'nmf':
                miner = NMFRoleMiner(**miner_params)
            elif algo == 'autoencoder':
                miner = AutoencoderRoleMiner(**miner_params)
            elif algo == 'gnn':
                miner = GNNRoleMiner(**miner_params)
            
            # Load and process
            miner.load_data(args.data)
            
            # Train model (quick mode for comparison)
            if algo == 'dbscan':
                miner.tune_parameters(metric=args.metric, plot=False)
                miner.fit(metric=args.metric)
            elif algo == 'kmeans':
                miner.tune_parameters(plot=False)
                miner.fit()
            elif algo == 'nmf':
                miner.tune_parameters(plot=False)
                miner.fit()
            elif algo == 'autoencoder':
                print("    Autoencoder training on CPU...")
                miner.fit()
            elif algo == 'gnn':
                print("    GNN building graph embeddings...")
                miner.fit()
            
            # Store results
            results[algo] = {
                'miner': miner,
                'metrics': miner.metrics,
                'roles': miner.roles,
                'display_name': display_name
            }
            
            print(f"   {display_name} completed")
            print(f"     Roles: {miner.metrics['num_roles']} | "
                  f"Silhouette: {miner.metrics['silhouette_score']:.3f} | "
                  f"Noise: {miner.metrics['noise_percentage']:.1f}%")
                  
        except Exception as e:
            print(f"   {display_name} failed: {e}")
            results[algo] = None
    
    # Display comparison table
    print("\n" + "="*80)
    print(" AI MODEL COMPARISON RESULTS")
    print("="*80)
    
    import pandas as pd
    
    comparison_data = []
    for algo, display_name in algorithms:
        if algo not in results or results[algo] is None:
            continue
            
        data = results[algo]
        m = data['metrics']
        
        row = {
            'Model': data['display_name'],
            'Roles': m['num_roles'],
            'Noise %': f"{m['noise_percentage']:.1f}%",
            'Avg Role Size': f"{m.get('avg_role_size', 0):.1f}",
            'Avg Perms/Role': f"{m.get('avg_permissions_per_role', 0):.1f}",
            'Silhouette': f"{m['silhouette_score']:.3f}",
            'Davies-Bouldin': f"{m['davies_bouldin_index']:.3f}",
            'Cohesion': f"{m.get('avg_cohesion', 0):.2%}"
        }
        
        # Add algorithm-specific metrics
        if algo == 'nmf' and hasattr(data['miner'], 'reconstruction_error'):
            row['Recon Error'] = f"{data['miner'].reconstruction_error:.4f}"
        elif algo == 'autoencoder' and hasattr(data['miner'], 'get_reconstruction_quality'):
            recon_quality = data['miner'].get_reconstruction_quality()
            row['Recon Acc'] = f"{recon_quality['binary_accuracy']:.2%}"
        elif algo == 'gnn' and hasattr(data['miner'], 'user_embeddings'):
            row['Embed Dim'] = f"{data['miner'].embedding_dim}"
        
        comparison_data.append(row)
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        
        # AI Model recommendations
        print("\n" + "="*80)
        print(" AI MODEL RECOMMENDATIONS")
        print("="*80)
        
        # Find best by silhouette score
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_silhouette = max(valid_results.items(), 
                                 key=lambda x: x[1]['metrics']['silhouette_score'])
            best_algo, best_data = best_silhouette
            print(f"\n BEST CLUSTERING QUALITY: {best_data['display_name']}")
            print(f"   Silhouette Score: {best_data['metrics']['silhouette_score']:.3f}")
            
            # Balanced recommendation (considering multiple factors)
            balanced_scores = {}
            for algo, data in valid_results.items():
                m = data['metrics']
                # Balance score: quality + coverage + reasonable role count
                score = (
                    m['silhouette_score'] * 100 +          # Clustering quality
                    (100 - m['noise_percentage']) +        # User coverage  
                    max(0, 20 - abs(m['num_roles'] - 10))  # Prefer ~10 roles
                )
                balanced_scores[algo] = score
            
            best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
            balanced_algo = valid_results[best_balanced[0]]['display_name']
            print(f" MOST BALANCED: {balanced_algo}")
            
            # Use case recommendations
            print(f"\n USE CASE GUIDANCE:")
            print(f"   • Quick PoC & Interpretability → 🚀 K-means")
            print(f"   • Auto-role discovery & Outliers → 🔍 DBSCAN") 
            print(f"   • Business-friendly roles → 📊 NMF")
            print(f"   • Complex patterns → 🧠 Autoencoder")
            print(f"   • Relationship learning → 🕸️ GNN")
            print(f"   • Your data works best with: {best_data['display_name']}")
    
    # Export comparison results
    if not args.no_export and comparison_data:
        print("\n" + "-"*80)
        print(" EXPORTING COMPARISON RESULTS")
        print("-"*80)
        
        # Export individual model results
        for algo, data in results.items():
            if data is None:
                continue
            output_dir = f"{args.output}/{algo}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            data['miner'].export_results(output_dir)
            print(f"   {data['display_name']} → {output_dir}/")
        
        # Export comparison table
        comparison_file = f"{args.output}/ai_model_comparison.csv"
        df_comparison.to_csv(comparison_file, index=False)
        print(f"   Comparison table → {comparison_file}")
        
        # Export recommendation
        if valid_results:
            with open(f"{args.output}/recommendation.txt", 'w') as f:
                f.write(f"AI ROLE MINING RECOMMENDATION\n")
                f.write("=" * 40 + "\n")
                f.write(f"Best Model: {best_data['display_name']}\n")
                f.write(f"Silhouette Score: {best_data['metrics']['silhouette_score']:.3f}\n")
                f.write(f"Roles Discovered: {best_data['metrics']['num_roles']}\n")
                f.write(f"Noise Percentage: {best_data['metrics']['noise_percentage']:.1f}%\n")
                f.write(f"Average Role Size: {best_data['metrics'].get('avg_role_size', 0):.1f}\n")
                f.write(f"Average Cohesion: {best_data['metrics'].get('avg_cohesion', 0):.2%}\n\n")
                f.write("Applied Thresholds:\n")
                f.write(f"  • Similarity: {args.similarity}\n")
                f.write(f"  • Grouping: {args.grouping}\n")
                f.write(f"  • Impact: {args.impact}\n")
            print(f"   Recommendation → {args.output}/recommendation.txt")
    
    print(f"\n AI MODEL COMPARISON COMPLETE!")
    print(f"   Detailed results in: {args.output}/")


if __name__ == "__main__":
    main()