# main.py

import os
# Disable GPU to avoid CUDA errors - critical for PoC!
# If a functioning GPU device is available, remove the following line
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import sys
from pathlib import Path

from dbscan_miner import DBSCANRoleMiner
from kmeans_miner import KMeansRoleMiner
from nmf_miner import NMFRoleMiner
from autoencoder_miner import AutoencoderRoleMiner

def main():
    parser = argparse.ArgumentParser(
        description=' AI Role Mining CLI - Intelligent Access Control Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Quick Start Examples:
  
  # Interactive mode (recommended)
  python main.py --data user_permissions.csv --interactive
  
  # Quick baseline with K-means
  python main.py --data user_permissions.csv --algorithm kmeans --plot
  
  # Smart role discovery with DBSCAN
  python main.py --data user_permissions.csv --algorithm dbscan --similarity 0.7
  
  # Interpretable roles with NMF
  python main.py --data user_permissions.csv --algorithm nmf --show-components
  
  # Advanced patterns with Autoencoder
  python main.py --data user_permissions.csv --algorithm autoencoder --epochs 50
  
  # Compare all AI models
  python main.py --data user_permissions.csv --compare --output results

Threshold Guide:
  --similarity (0.6-0.8) : How similar users must be (higher = stricter)
  --grouping   (3-10)    : Minimum users per role  
  --impact     (0.7-0.9) : Permission inclusion threshold
        """
    )
    
    # Required arguments
    parser.add_argument('--data', required=True,
                       help='Path to user-permission matrix (CSV/XLSX)')
    
    # Algorithm selection
    parser.add_argument('--algorithm', 
                       choices=['kmeans', 'dbscan', 'nmf', 'autoencoder'],
                       help='AI model for role mining')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare all AI models and recommend best')
    
    parser.add_argument('--interactive', action='store_true',
                       help=' Interactive PoC mode with guided setup')
    
    # Core threshold parameters
    parser.add_argument('--similarity', type=float, default=0.7,
                       help='User similarity threshold (0-1, default: 0.7)')
    
    parser.add_argument('--grouping', type=int, default=5,
                       help='Minimum users per role (default: 5)')
    
    parser.add_argument('--impact', type=float, default=0.8,
                       help='Permission inclusion threshold (0-1, default: 0.8)')
    
    # Algorithm-specific parameters
    parser.add_argument('--n-clusters', type=int,
                       help='Number of roles (K-means)')
    
    parser.add_argument('--n-components', type=int,
                       help='Number of components (NMF)')
    
    parser.add_argument('--encoding-dim', type=int, default=32,
                       help='Latent dimension for autoencoder (default: 32)')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs for autoencoder (default: 100)')
    
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
    
    # Interactive PoC mode
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
    print("   5. Compare All Models (Find optimal approach)")
    
    choice = input("\nSelect approach (1-5): ").strip()
    
    use_case_map = {
        '1': ('kmeans', " Quick Baseline - K-means"),
        '2': ('dbscan', " Smart Discovery - DBSCAN"), 
        '3': ('nmf', " Interpretable Roles - NMF"),
        '4': ('autoencoder', " Advanced Patterns - Autoencoder"),
        '5': ('compare', " Compare All AI Models")
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
    if algo != 'compare':
        print(f"\n  CONFIGURATION FOR {description.split(' - ')[1]}")
    
    if algo == 'kmeans':
        print("   K-means works best with 8-15 roles for most organizations")
        clusters = input("   How many roles to discover? (press Enter for auto-tune): ").strip()
        if clusters:
            args.n_clusters = int(clusters)
    
    elif algo == 'dbscan':
        print("   DBSCAN auto-detects roles based on density")
        print("   • Similarity: How close users must be to form roles")
        sim = input(f"   Similarity threshold (0.5-0.9, Enter for {args.similarity}): ").strip()
        if sim:
            args.similarity = float(sim)
    
    elif algo == 'nmf': 
        print("   NMF discovers interpretable role components")
        print("   • Impact: How common permissions must be to include in roles")
        impact = input(f"   Impact threshold (0.6-0.9, Enter for {args.impact}): ").strip()
        if impact:
            args.impact = float(impact)
    
    elif algo == 'autoencoder':
        print("   Autoencoder learns complex permission patterns")
        print("   • Training: More epochs = better patterns (but slower)")
        epochs = input(f"   Training epochs (50-200, Enter for {args.epochs}): ").strip()
        if epochs:
            args.epochs = int(epochs)
    
    # Step 4: Quick thresholds
    print(f"\n QUICK THRESHOLDS")
    
    group = input(f"   Minimum users per role (3-10, Enter for {args.grouping}): ").strip()
    if group:
        args.grouping = int(group)
    
    # Step 5: Output options
    print(f"\n OUTPUT OPTIONS")
    args.plot = input("   Generate visualizations? (y/n, default y): ").strip().lower() != 'n'
    
    if algo in ['nmf']:
        args.show_components = input("   Show role components? (y/n, default y): ").strip().lower() != 'n'
    elif algo == 'autoencoder':
        args.show_latent = input("   Show latent space? (y/n, default y): ").strip().lower() != 'n'
    
    output_dir = input(f"   Output directory (Enter for '{args.output}'): ").strip()
    if output_dir:
        args.output = output_dir
    
    # Summary
    print(f"\n PoC CONFIGURATION SUMMARY")
    print(f"   • Approach: {description}")
    print(f"   • Data: {Path(args.data).name}")
    print(f"   • Grouping: Min {args.grouping} users per role")
    print(f"   • Visualizations: {'Yes' if args.plot else 'No'}")
    print(f"   • Output: {args.output}/")
    
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
    
    # Initialize the selected AI miner
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
        algo_name = " Deep Autoencoder"
    
    print(f"\n{algo_name}")
    print(f"   Similarity: {args.similarity} | Grouping: {args.grouping} | Impact: {args.impact}")
    
    # Load and analyze data
    print(f"\n LOADING DATA...")
    data = miner.load_data(args.data)
    
    # Algorithm-specific tuning
    print(f"\n  AI MODEL OPTIMIZATION...")
    
    if args.algorithm == 'kmeans':
        miner.tune_parameters(
            plot=args.plot,
            save_plot=f'{args.output}/kmeans_tuning.png' if args.plot else None
        )
        fit_params = {'n_clusters': args.n_clusters} if args.n_clusters else {}
        
    elif args.algorithm == 'dbscan':
        miner.tune_parameters(
            metric=args.metric,
            plot=args.plot, 
            save_plot=f'{args.output}/dbscan_tuning.png' if args.plot else None
        )
        fit_params = {'metric': args.metric}
        
    elif args.algorithm == 'nmf':
        miner.tune_parameters(
            plot=args.plot,
            save_plot=f'{args.output}/nmf_tuning.png' if args.plot else None
        )
        fit_params = {'n_components': args.n_components} if args.n_components else {}
        
    elif args.algorithm == 'autoencoder':
        # Autoencoder tunes during fit
        fit_params = {'n_clusters': args.n_clusters} if args.n_clusters else {}
        print("   Autoencoder will tune during training...")
    
    # Train the AI model
    print(f"\n TRAINING AI MODEL...")
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
    
    print(f"\n {algo_name} COMPLETED!")
    print(f"   Results saved to: {args.output}/")


def compare_ai_models(args):
    """Compare all AI models and recommend best approach"""
    print("\n" + "="*80)
    print(" COMPARING AI ROLE MINING MODELS")
    print("="*80)
    
    algorithms = [
        ('kmeans', ' K-means'),
        ('dbscan', ' DBSCAN'), 
        ('nmf', ' NMF'),
        ('autoencoder', ' Autoencoder')
    ]
    
    results = {}
    
    for algo, display_name in algorithms:
        print(f"\n{'─'*80}")
        print(f"Testing: {display_name}")
        print('─'*80)
        
        try:
            # Initialize miner
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
            
            # Load data
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
                miner.fit()  # Uses internal tuning
            
            # Store results
            results[algo] = {
                'miner': miner,
                'metrics': miner.metrics,
                'display_name': display_name
            }
            
            print(f"   {display_name} completed")
            print(f"     Roles: {miner.metrics['num_roles']} | "
                  f"Silhouette: {miner.metrics['silhouette_score']:.3f} | "
                  f"Noise: {miner.metrics['noise_percentage']:.1f}%")
                  
        except Exception as e:
            print(f"   {display_name} failed: {e}")
            results[algo] = None
    
    # Display comparison results
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
        
        # Add AI-specific metrics
        if algo == 'nmf' and hasattr(data['miner'], 'reconstruction_error'):
            row['Recon Error'] = f"{data['miner'].reconstruction_error:.4f}"
        elif algo == 'autoencoder' and hasattr(data['miner'], 'get_reconstruction_quality'):
            recon_quality = data['miner'].get_reconstruction_quality()
            row['Recon Acc'] = f"{recon_quality['binary_accuracy']:.2%}"
        
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
            print(f"   • Quick PoC & Interpretability →  K-means")
            print(f"   • Auto-role discovery & Outliers →  DBSCAN") 
            print(f"   • Business-friendly roles →  NMF")
            print(f"   • Complex patterns & Best accuracy →  Autoencoder")
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
        with open(f"{args.output}/recommendation.txt", 'w') as f:
            f.write(f"AI ROLE MINING RECOMMENDATION\n")
            f.write(f"Best Model: {best_data['display_name']}\n")
            f.write(f"Silhouette Score: {best_data['metrics']['silhouette_score']:.3f}\n")
            f.write(f"Roles Discovered: {best_data['metrics']['num_roles']}\n")
        print(f"   Recommendation → {args.output}/recommendation.txt")
    
    print(f"\n AI MODEL COMPARISON COMPLETE!")
    print(f"   Detailed results in: {args.output}/")


if __name__ == "__main__":
    main()