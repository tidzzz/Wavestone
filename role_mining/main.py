# main.py

import argparse
import sys
from pathlib import Path

from dbscan_miner import DBSCANRoleMiner
from kmeans_miner import KMeansRoleMiner

def main():
    parser = argparse.ArgumentParser(
        description=' Role Mining CLI - IAM Access Control Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python main.py --data out/users_permission_matrix.csv --interactive
  
  # DBSCAN with custom thresholds
  python main.py --data out/users_permission_matrix.csv --algorithm dbscan \\
      --similarity 0.7 --grouping 5 --impact 0.8
  
  # K-means auto-tuned
  python main.py --data out/users_permission_matrix.csv --algorithm kmeans

  
  # Compare all algorithms
  python main.py --data out/users_permission_matrix.csv --compare \\
      --similarity 0.7 --grouping 3 --impact 0.8
  
  # Adjust thresholds after clustering
  python main.py --data out/users_permission_matrix.csv --algorithm dbscan \\
      --similarity 0.7 --adjust-impact 0.9

Threshold Parameters:
  --similarity   : User-user similarity (0-1, higher=stricter grouping)
  --grouping     : Minimum users per role (int, e.g., 3, 5, 10)
  --impact       : Permission inclusion (0-1, % of users needing perm)
        """
    )
    
    # Required
    parser.add_argument('--data', required=True,
                       help='Path to user-permission matrix (CSV/XLSX)')
    
    # Algorithm
    parser.add_argument('--algorithm', choices=['dbscan', 'kmeans'],
                       help='Clustering algorithm')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare all algorithms')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode with prompts')
    
    # Threshold parameters (the important ones!)
    parser.add_argument('--similarity', type=float, default=0.7,
                       help='Similarity threshold (0-1, default: 0.7)')
    
    parser.add_argument('--grouping', type=int, default=3,
                       help='Grouping threshold - min users per role (default: 3)')
    
    parser.add_argument('--impact', type=float, default=0.8,
                       help='Impact threshold - permission inclusion (0-1, default: 0.8)')
    
    # Algorithm-specific
    parser.add_argument('--n-clusters', type=int,
                       help='Number of clusters (K-means)')
    
    parser.add_argument('--metric', default='jaccard',
                       choices=['jaccard', 'cosine', 'euclidean'],
                       help='Distance metric for DBSCAN')
    
    # Post-processing
    parser.add_argument('--handle-noise', 
                       choices=['assign_nearest', 'create_micro_roles', 'flag_for_review'],
                       help='How to handle noise/rejected users')
    
    parser.add_argument('--adjust-impact', type=float,
                       help='Re-extract roles with different impact threshold')
    
    # Visualization
    parser.add_argument('--plot', action='store_true',
                       help='Show plots (tuning, clustering)')
    
    # Output
    parser.add_argument('--output', default='results',
                       help='Output directory (default: results)')
    
    parser.add_argument('--no-export', action='store_true',
                       help='Skip exporting results')
    
    args = parser.parse_args()
    
    # Validate data file
    if not Path(args.data).exists():
        print(f" Error: File not found: {args.data}")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        run_interactive(args)
        return
    
    # Compare mode
    if args.compare:
        compare_algorithms(args)
        return
    
    # Single algorithm mode
    if not args.algorithm:
        print(" Error: Specify --algorithm or use --compare or --interactive")
        parser.print_help()
        sys.exit(1)
    
    run_single_algorithm(args)


def run_interactive(args):
    """Interactive mode with prompts"""
    print("\n" + "="*80)
    print(" INTERACTIVE ROLE MINING")
    print("="*80)
    
    # Step 1: Choose algorithm
    print("\nAvailable algorithms:")
    print("  1. DBSCAN       - Auto-detects number of roles, handles outliers")
    print("  2. K-means      - Fast, requires number of roles")
    print("  3. Compare all  - Test all algorithms")
    
    choice = input("\nSelect (1-3): ").strip()
    
    algo_map = {'1': 'dbscan', '2': 'kmeans','3': 'compare'}
    if choice not in algo_map:
        print(" Invalid choice")
        return
    
    if choice == '3':
        args.compare = True
    else:
        args.algorithm = algo_map[choice]
    
    # Step 2: Configure thresholds
    print("\n" + "-"*80)
    print("  THRESHOLD CONFIGURATION")
    print("-"*80)
    
    print(f"\n1. Similarity Threshold (current: {args.similarity})")
    print("   How similar must users be to form a role?")
    print("   • 0.5 = Loose (more diverse roles)")
    print("   • 0.7 = Balanced (recommended)")
    print("   • 0.9 = Strict (very homogeneous roles)")
    sim_input = input(f"   Enter value (or press Enter for {args.similarity}): ").strip()
    if sim_input:
        args.similarity = float(sim_input)
    
    print(f"\n2. Grouping Threshold (current: {args.grouping})")
    print("   Minimum number of users to form a valid role")
    print("   • 3 = Allow small specialized roles")
    print("   • 5-10 = Standard (recommended)")
    print("   • 20+ = Only large, common roles")
    grp_input = input(f"   Enter value (or press Enter for {args.grouping}): ").strip()
    if grp_input:
        args.grouping = int(grp_input)
    
    print(f"\n3. Impact Threshold (current: {args.impact})")
    print("   % of users that must have a permission to include it in role")
    print("   • 0.6 = Include permissions used by 60%+ of users")
    print("   • 0.8 = Standard (recommended)")
    print("   • 1.0 = Only permissions shared by ALL users")
    imp_input = input(f"   Enter value (or press Enter for {args.impact}): ").strip()
    if imp_input:
        args.impact = float(imp_input)
    
    print(f"\n Configuration:")
    print(f"   • Similarity: {args.similarity}")
    print(f"   • Grouping: {args.grouping}")
    print(f"   • Impact: {args.impact}")
    
    input("\nPress Enter to start clustering...")
    
    # Run
    if args.compare:
        compare_algorithms(args)
    else:
        run_single_algorithm(args)


def run_single_algorithm(args):
    """Run a single clustering algorithm"""
    print("\n" + "="*80)
    print(f" RUNNING {args.algorithm.upper()}")
    print("="*80)
    
    # Initialize miner
    if args.algorithm == 'dbscan':
        miner = DBSCANRoleMiner(
            similarity_threshold=args.similarity,
            grouping_threshold=args.grouping,
            impact_threshold=args.impact
        )
    elif args.algorithm == 'kmeans':
        miner = KMeansRoleMiner(
            similarity_threshold=args.similarity,
            grouping_threshold=args.grouping,
            impact_threshold=args.impact
        )
    
    # Load data
    miner.load_data(args.data)
    
    # Tune parameters
    print("\n" + "-"*80)
    print(" PARAMETER TUNING")
    print("-"*80)
    
    if args.algorithm == 'dbscan':
        miner.tune_parameters(
            metric=args.metric,
            plot=args.plot,
            save_plot=f'{args.output}/dbscan_tuning.png' if args.plot else None
        )
    elif args.algorithm == 'kmeans':
        miner.tune_parameters(
            plot=args.plot,
            save_plot=f'{args.output}/kmeans_tuning.png' if args.plot else None
        )
    
    # Fit model
    print("\n" + "-"*80)
    print("  CLUSTERING")
    print("-"*80)
    
    if args.algorithm == 'dbscan':
        miner.fit(metric=args.metric)
    elif args.algorithm == 'kmeans':
        miner.fit(n_clusters=args.n_clusters)
    # Display results
    miner.print_summary()
    
    # Adjust impact threshold if requested
    if args.adjust_impact:
        print("\n" + "-"*80)
        print(f" ADJUSTING IMPACT THRESHOLD TO {args.adjust_impact}")
        print("-"*80)
        miner.adjust_thresholds(impact_threshold=args.adjust_impact)
        miner.print_summary()
    
    # Handle noise
    if args.handle_noise:
        print("\n" + "-"*80)
        print(" HANDLING NOISE/REJECTED USERS")
        print("-"*80)
        if args.algorithm == 'dbscan':
            miner.handle_noise(strategy=args.handle_noise)
        else:
            print("  Noise handling only available for DBSCAN")
    
    # Visualizations
    if args.plot:
        print("\n" + "-"*80)
        print(" GENERATING VISUALIZATIONS")
        print("-"*80)
        
        miner.visualize_results(
            save_path=f'{args.output}/{args.algorithm}_clusters.png'
        )
    
    # Export results
    if not args.no_export:
        print("\n" + "-"*80)
        print(" EXPORTING RESULTS")
        print("-"*80)
        Path(args.output).mkdir(exist_ok=True)
        miner.export_results(args.output)
    
    print("\n Done!")


def compare_algorithms(args):
    """Compare all three algorithms"""
    print("\n" + "="*80)
    print(" COMPARING ALL ALGORITHMS")
    print("="*80)
    
    algorithms = ['dbscan', 'kmeans']
    results = {}
    
    for algo in algorithms:
        print(f"\n{'─'*80}")
        print(f"Testing: {algo.upper()}")
        print('─'*80)
        
        # Initialize
        if algo == 'dbscan':
            miner = DBSCANRoleMiner(
                similarity_threshold=args.similarity,
                grouping_threshold=args.grouping,
                impact_threshold=args.impact
            )
        elif algo == 'kmeans':
            miner = KMeansRoleMiner(
                similarity_threshold=args.similarity,
                grouping_threshold=args.grouping,
                impact_threshold=args.impact
            )

        # Load and process
        miner.load_data(args.data)
        
        # Tune (no plots for comparison)
        if algo == 'dbscan':
            miner.tune_parameters(metric=args.metric, plot=False)
            miner.fit(metric=args.metric)
        elif algo == 'kmeans':
            miner.tune_parameters(plot=False)
            miner.fit()

        # Store results
        results[algo] = {
            'miner': miner,
            'metrics': miner.metrics,
            'roles': miner.roles
        }
    
    # Display comparison table
    print("\n" + "="*80)
    print(" COMPARISON RESULTS")
    print("="*80)
    
    import pandas as pd
    
    comparison_data = []
    for algo, data in results.items():
        m = data['metrics']
        comparison_data.append({
            'Algorithm': algo.upper(),
            'Roles': m['num_roles'],
            'Noise %': f"{m['noise_percentage']:.1f}%",
            'Avg Role Size': f"{m.get('avg_role_size', 0):.1f}",
            'Avg Perms/Role': f"{m.get('avg_permissions_per_role', 0):.1f}",
            'Silhouette': f"{m['silhouette_score']:.3f}",
            'Davies-Bouldin': f"{m['davies_bouldin_index']:.3f}",
            'Avg Cohesion': f"{m.get('avg_cohesion', 0):.2%}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    # Best algorithm recommendation
    print("\n" + "="*80)
    print(" RECOMMENDATIONS")
    print("="*80)
    
    # Find best by silhouette
    best_silhouette = max(results.items(), 
                         key=lambda x: x[1]['metrics']['silhouette_score'])
    print(f"\n Best clustering quality (Silhouette): {best_silhouette[0].upper()}")
    
    # Find most balanced
    balanced_scores = {}
    for algo, data in results.items():
        m = data['metrics']
        # Balance: good silhouette, low noise, reasonable number of roles
        score = (
            m['silhouette_score'] * 100 +
            (100 - m['noise_percentage']) -
            abs(m['num_roles'] - 10) * 2  # Penalty for being far from 10 roles
        )
        balanced_scores[algo] = score
    
    best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
    print(f"  Most balanced overall: {best_balanced[0].upper()}")
    
    # Recommendations by use case
    print(f"\n Use case recommendations:")
    print(f"   • Unknown number of roles → DBSCAN")
    print(f"   • Need specific number of roles → K-means")
    print(f"   • Have many outliers → DBSCAN")
    print(f"   • Need fast results → K-means")
    
    # Export all results
    if not args.no_export:
        print("\n" + "-"*80)
        print(" EXPORTING ALL RESULTS")
        print("-"*80)
        
        for algo, data in results.items():
            output_dir = f"{args.output}/{algo}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            data['miner'].export_results(output_dir)
        
        # Export comparison table
        comparison_file = f"{args.output}/comparison.csv"
        df_comparison.to_csv(comparison_file, index=False)
        print(f" Comparison table → {comparison_file}")
    
    print("\n Comparison complete!")


if __name__ == "__main__":
    main()