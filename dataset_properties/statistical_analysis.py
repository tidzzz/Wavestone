"""
iam_statistical_analysis.py
Analyse statistique du dataset IAM pour le Role Mining

Dépendances : pip install pandas numpy scipy
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import json

INPUT_DIR = "iam_dataset"


def load_data():
    """Charge les fichiers CSV du dataset IAM"""
    users = pd.read_csv(f"{INPUT_DIR}/users.csv")
    apps = pd.read_csv(f"{INPUT_DIR}/applications.csv")
    perms = pd.read_csv(f"{INPUT_DIR}/permissions.csv")
    rights = pd.read_csv(f"{INPUT_DIR}/rights.csv")
    return users, apps, perms, rights


def print_section(title):
    """Affiche un titre de section"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)


def calculate_basic_stats(users, apps, perms, rights):
    """Calcule les statistiques de base du dataset"""
    stats_dict = {
        'n_users': len(users),
        'n_applications': len(apps),
        'n_permissions': len(perms),
        'n_rights': len(rights)
    }
    return stats_dict


def calculate_matrix_metrics(n_users, n_perms, n_rights):
    """Calcule les métriques de la matrice User-Permission"""
    max_possible = n_users * n_perms
    density = (n_rights / max_possible) * 100
    sparsity = 100 - density
    
    metrics = {
        'matrix_dimensions': f"{n_users:,} × {n_perms:,}",
        'max_possible_rights': max_possible,
        'actual_rights': n_rights,
        'density_percent': density,
        'sparsity_percent': sparsity,
        'compression_factor': max_possible / n_rights
    }
    return metrics


def calculate_permissions_per_user_stats(rights):
    """Calcule les statistiques des permissions par utilisateur"""
    perms_per_user = rights.groupby('user_id').size()
    
    stats_dict = {
        'mean': perms_per_user.mean(),
        'median': perms_per_user.median(),
        'std': perms_per_user.std(),
        'min': perms_per_user.min(),
        'max': perms_per_user.max(),
        'q1': perms_per_user.quantile(0.25),
        'q3': perms_per_user.quantile(0.75),
        'coefficient_variation_percent': (perms_per_user.std() / perms_per_user.mean()) * 100
    }
    
    # Test de normalité
    sample = perms_per_user.sample(min(5000, len(perms_per_user)))
    _, p_value_shapiro = stats.shapiro(sample)
    stats_dict['shapiro_p_value'] = p_value_shapiro
    stats_dict['is_normal_distribution'] = p_value_shapiro >= 0.05
    
    # Asymétrie et aplatissement
    stats_dict['skewness'] = stats.skew(perms_per_user)
    stats_dict['kurtosis'] = stats.kurtosis(perms_per_user)
    
    return stats_dict, perms_per_user


def calculate_users_per_permission_stats(rights, n_perms):
    """Calcule les statistiques des utilisateurs par permission"""
    users_per_perm = rights.groupby('permission_id').size()
    
    stats_dict = {
        'mean': users_per_perm.mean(),
        'median': users_per_perm.median(),
        'std': users_per_perm.std(),
        'min': users_per_perm.min(),
        'max': users_per_perm.max(),
        'q1': users_per_perm.quantile(0.25),
        'q3': users_per_perm.quantile(0.75)
    }
    
    # Classification des permissions
    rare_threshold = users_per_perm.quantile(0.25)
    common_threshold = users_per_perm.quantile(0.75)
    
    stats_dict['rare_permissions_count'] = (users_per_perm < rare_threshold).sum()
    stats_dict['common_permissions_count'] = (users_per_perm > common_threshold).sum()
    stats_dict['rare_permissions_percent'] = (stats_dict['rare_permissions_count'] / n_perms) * 100
    stats_dict['common_permissions_percent'] = (stats_dict['common_permissions_count'] / n_perms) * 100
    
    return stats_dict, users_per_perm


def calculate_business_attributes_stats(users, n_users):
    """Calcule les statistiques des attributs métier"""
    stats_dict = {
        'n_positions': users['position'].nunique(),
        'n_departments': users['department'].nunique(),
        'n_locations': users['location'].nunique(),
        'n_contract_types': users['contract_type'].nunique(),
        'n_seniority_levels': users['seniority'].nunique()
    }
    
    # Distributions en pourcentage
    stats_dict['position_distribution'] = {
        pos: {'count': count, 'percent': (count/n_users)*100}
        for pos, count in users['position'].value_counts().items()
    }
    stats_dict['department_distribution'] = {
        dept: {'count': count, 'percent': (count/n_users)*100}
        for dept, count in users['department'].value_counts().items()
    }
    stats_dict['location_distribution'] = {
        loc: {'count': count, 'percent': (count/n_users)*100}
        for loc, count in users['location'].value_counts().items()
    }
    stats_dict['contract_distribution'] = {
        ct: {'count': count, 'percent': (count/n_users)*100}
        for ct, count in users['contract_type'].value_counts().items()
    }
    stats_dict['seniority_distribution'] = {
        sen: {'count': count, 'percent': (count/n_users)*100}
        for sen, count in users['seniority'].value_counts().items()
    }
    
    return stats_dict


def calculate_similarity_metrics(rights, sample_size=1000, pairs_per_user=20):
    """Calcule les métriques de similarité Jaccard entre utilisateurs"""
    all_users = rights['user_id'].unique()
    sampled_users = np.random.choice(all_users, min(sample_size, len(all_users)), replace=False)
    
    # Construction des ensembles de permissions par utilisateur
    user_permissions = defaultdict(set)
    for _, row in rights[rights['user_id'].isin(sampled_users)].iterrows():
        user_permissions[row['user_id']].add(row['permission_id'])
    
    # Calcul des similarités Jaccard
    similarities = []
    users_list = list(user_permissions.keys())
    
    for i in range(min(500, len(users_list))):
        for j in range(i + 1, min(i + pairs_per_user, len(users_list))):
            perms1 = user_permissions[users_list[i]]
            perms2 = user_permissions[users_list[j]]
            
            intersection = len(perms1 & perms2)
            union = len(perms1 | perms2)
            
            if union > 0:
                jaccard = intersection / union
                similarities.append(jaccard)
    
    if not similarities:
        return None
    
    similarity_stats = {
        'mean': np.mean(similarities),
        'median': np.median(similarities),
        'std': np.std(similarities),
        'min': np.min(similarities),
        'max': np.max(similarities),
        'q1': np.percentile(similarities, 25),
        'q3': np.percentile(similarities, 75),
        'sample_size': len(sampled_users),
        'pairs_evaluated': len(similarities)
    }
    
    return similarity_stats


def calculate_entropy_metrics(users_per_perm):
    """Calcule l'entropie et les métriques de complexité"""
    # Entropie de Shannon
    perm_probs = users_per_perm / users_per_perm.sum()
    entropy = -np.sum(perm_probs * np.log2(perm_probs + 1e-10))
    max_entropy = np.log2(len(users_per_perm))
    normalized_entropy = entropy / max_entropy
    
    # Coefficient de Gini
    sorted_values = np.sort(users_per_perm.values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((n - np.arange(1, n + 1) + 1) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
    
    metrics = {
        'shannon_entropy_bits': entropy,
        'max_entropy_bits': max_entropy,
        'normalized_entropy': normalized_entropy,
        'gini_coefficient': gini
    }
    
    return metrics


def calculate_application_stats(rights, perms, apps):
    """Calcule les statistiques par application"""    
    users_per_app = rights.groupby('application_id')['user_id'].nunique()
    perms_per_app = perms.groupby('application_id').size()
    
    stats_dict = {
        'apps_used': len(users_per_app),
        'apps_total': len(apps),
        'apps_unused': len(apps) - len(users_per_app),
        'mean_users_per_app': users_per_app.mean(),
        'median_users_per_app': users_per_app.median(),
        'std_users_per_app': users_per_app.std(),
        'min_users_per_app': users_per_app.min(),
        'max_users_per_app': users_per_app.max(),
        'mean_permissions_per_app': perms_per_app.mean(),
        'median_permissions_per_app': perms_per_app.median(),
        'std_permissions_per_app': perms_per_app.std()
    }
    
    return stats_dict


def print_statistics(data, indent=2):
    """Affiche les statistiques de manière formatée"""
    indent_str = " " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            # Ne pas afficher les distributions complètes
            if 'distribution' in key:
                continue
            print(f"{indent_str}{key}:")
            print_statistics(value, indent + 2)
        elif isinstance(value, float):
            print(f"{indent_str}{key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"{indent_str}{key}: {value:,}")
        elif isinstance(value, bool):
            print(f"{indent_str}{key}: {value}")
        else:
            print(f"{indent_str}{key}: {value}")



def main():
    print("="*80)
    print("ANALYSE STATISTIQUE IAM - ROLE MINING")
    print("="*80)
    
    # Chargement des données
    print("\n[Chargement des données...]")
    users, apps, perms, rights = load_data()
    
    # 1. Statistiques de base
    print_section("1. STATISTIQUES DE BASE")
    basic_stats = calculate_basic_stats(users, apps, perms, rights)
    print_statistics(basic_stats)
    
    # 2. Métriques de la matrice
    print_section("2. MÉTRIQUES MATRICE USER-PERMISSION")
    matrix_metrics = calculate_matrix_metrics(
        basic_stats['n_users'],
        basic_stats['n_permissions'],
        basic_stats['n_rights']
    )
    print_statistics(matrix_metrics)
    
    # 3. Permissions par utilisateur
    print_section("3. PERMISSIONS PAR UTILISATEUR")
    perms_user_stats, perms_per_user = calculate_permissions_per_user_stats(rights)
    print_statistics(perms_user_stats)
    
    # 4. Utilisateurs par permission
    print_section("4. UTILISATEURS PAR PERMISSION")
    users_perm_stats, users_per_perm = calculate_users_per_permission_stats(
        rights, 
        basic_stats['n_permissions']
    )
    print_statistics(users_perm_stats)
    
    # 5. Attributs métier
    print_section("5. ATTRIBUTS MÉTIER")
    business_stats = calculate_business_attributes_stats(users, basic_stats['n_users'])
    
    print(f"  n_positions: {business_stats['n_positions']}")
    print(f"  n_departments: {business_stats['n_departments']}")
    print(f"  n_locations: {business_stats['n_locations']}")
    print(f"  n_contract_types: {business_stats['n_contract_types']}")
    print(f"  n_seniority_levels: {business_stats['n_seniority_levels']}")
    
    print("\n  Top 5 Positions:")
    for i, (pos, data) in enumerate(list(business_stats['position_distribution'].items())[:5], 1):
        print(f"    {i}. {pos}: {data['count']} ({data['percent']:.1f}%)")
    
    print("\n  Locations:")
    for loc, data in business_stats['location_distribution'].items():
        print(f"    {loc}: {data['count']} ({data['percent']:.1f}%)")
    
    # 6. Similarité
    print_section("6. SIMILARITÉ JACCARD (échantillon)")
    similarity_stats = calculate_similarity_metrics(rights)
    if similarity_stats:
        print_statistics(similarity_stats)
    else:
        print("  Calcul impossible (échantillon trop petit)")
    
    # 7. Entropie et complexité
    print_section("7. ENTROPIE ET COMPLEXITÉ")
    entropy_metrics = calculate_entropy_metrics(users_per_perm)
    print_statistics(entropy_metrics)
    
    # 8. Statistiques par application
    print_section("8. STATISTIQUES PAR APPLICATION")
    app_stats = calculate_application_stats(rights, perms, apps)
    print_statistics(app_stats)
    
    # Export JSON
    print_section("EXPORT")
    
    export_summary = {}
    
    # Basic stats
    export_summary.update(basic_stats)
    
    # Matrix metrics
    export_summary.update({f'matrix_{k}': v for k, v in matrix_metrics.items()})
    
    # Permissions per user
    export_summary.update({f'perms_per_user_{k}': v for k, v in perms_user_stats.items()})
    
    # Users per permission
    export_summary.update({f'users_per_perm_{k}': v for k, v in users_perm_stats.items()})
    
    # Business attributes (counts only)
    export_summary['n_positions'] = business_stats['n_positions']
    export_summary['n_departments'] = business_stats['n_departments']
    export_summary['n_locations'] = business_stats['n_locations']
    export_summary['n_contract_types'] = business_stats['n_contract_types']
    export_summary['n_seniority_levels'] = business_stats['n_seniority_levels']
    
    # Similarity
    if similarity_stats:
        export_summary.update({f'similarity_{k}': v for k, v in similarity_stats.items()})
    
    # Entropy
    export_summary.update(entropy_metrics)
    
    # Application stats
    export_summary.update({f'app_{k}': v for k, v in app_stats.items()})
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if hasattr(obj, 'item'):
            return obj.item()  # Convert numpy types to native Python types
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Apply conversion to the entire export_summary
    export_summary_serializable = convert_to_serializable(export_summary)
    
    # Save to JSON
    output_file = f"{INPUT_DIR}/statistical_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_summary_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Fichier: {output_file}")
    print(f"  Métriques exportées: {len(export_summary)}")
    
    print("\n" + "="*80)
    print("ANALYSE TERMINÉE")
    print("="*80)


if __name__ == "__main__":
    main()