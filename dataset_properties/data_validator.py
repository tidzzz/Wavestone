"""
data_validator.py
IAM dataset validator with corrected building access validation
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
from datetime import datetime

class IAMDatasetValidator:
    """
    Comprehensive validator for IAM dataset quality and AI readiness
    """
    
    def __init__(self, data_dir="iam_dataset", config=None):
        self.data_dir = data_dir
        self.config = config or self._default_config()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_suites': {},
            'overall_score': 0,
            'recommendations': []
        }
        
        # Data storage
        self.users = None
        self.applications = None
        self.permissions = None
        self.rights = None
        
    def _default_config(self):
        """Default validation thresholds"""
        return {
            'min_users': 5000,
            'min_rights': 10000,
            'consistency_threshold': 0.7,
            'entropy_threshold': 1.0,
            'sparsity_threshold': 0.99,
            'balance_threshold': 0.01,
            'cv_min': 0.3,
            'cv_max': 1.0
        }
    
    def load_datasets(self):
        """Load all dataset files into instance state"""
        try:
            self.users = pd.read_csv(f"{self.data_dir}/users.csv")
            self.applications = pd.read_csv(f"{self.data_dir}/applications.csv")
            self.permissions = pd.read_csv(f"{self.data_dir}/permissions.csv")
            self.rights = pd.read_csv(f"{self.data_dir}/rights.csv")
            
            print("✓ Datasets loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading datasets: {e}")
            return False
    
    def run_comprehensive_validation(self):
        """Run all validation suites"""
        if not self.load_datasets():
            return False
        
        print("🔍 Starting Comprehensive IAM Dataset Validation")
        print("=" * 60)
        
        # Run all test suites
        test_suites = [
            self.validate_data_completeness,
            self.validate_business_rules, 
            self.validate_ai_readiness,
            self.validate_data_patterns,
        ]
        
        for test_suite in test_suites:
            suite_name = test_suite.__name__.replace('validate_', '').replace('_', ' ').title()
            print(f"\n📋 Running {suite_name}...")
            self.results['test_suites'][test_suite.__name__] = test_suite()
        
        self._calculate_overall_score()
        self._generate_recommendations()
        self.generate_report()
        
        return self.results['overall_score'] >= 70
    
    def validate_data_completeness(self):
        """Test suite for data integrity and completeness"""
        tests = []
        
        # Test 1: Missing values
        for df_name, df in [
            ("users", self.users), 
            ("applications", self.applications),
            ("permissions", self.permissions), 
            ("rights", self.rights)
        ]:
            missing = df.isnull().sum().sum()
            tests.append({
                'name': f'No missing values in {df_name}',
                'passed': missing == 0,
                'details': f'{missing} missing values found' if missing > 0 else 'No missing values'
            })
        
        # Test 2: Foreign key integrity
        fk_tests = [
            ('user_id', self.users['user_id'], "users"),
            ('application_id', self.applications['application_id'], "applications"), 
            ('permission_id', self.permissions['permission_id'], "permissions")
        ]
        
        for fk_name, valid_values, source_table in fk_tests:
            if fk_name in self.rights.columns:
                invalid_refs = self.rights[~self.rights[fk_name].isin(valid_values)]
                tests.append({
                    'name': f'Foreign key integrity: {fk_name} -> {source_table}',
                    'passed': len(invalid_refs) == 0,
                    'details': f'{len(invalid_refs)} invalid references found' if len(invalid_refs) > 0 else 'All references valid'
                })
        
        # Test 3: Expected data volumes
        tests.extend([
            {
                'name': f'User count meets minimum ({self.config["min_users"]})',
                'passed': len(self.users) >= self.config['min_users'],
                'details': f'Found {len(self.users)} users'
            },
            {
                'name': f'Sufficient permission assignments ({self.config["min_rights"]})',
                'passed': len(self.rights) >= self.config['min_rights'],
                'details': f'Found {len(self.rights)} permission assignments'
            }
        ])
        
        return self._score_test_suite(tests, "Data Completeness")
    
    def validate_business_rules(self):
        """Validate business logic and domain rules - FIXED VERSION"""
        tests = []
        
        # Test: Building access by location - CORRECTED
        building_access_issues = 0
        
        for location in self.users['location'].unique():
            location_users = self.users[self.users['location'] == location]
            
            # Find the badge application for this location
            badge_app_name = f"Access Badge System - {location}"
            badge_apps = self.applications[self.applications['name'] == badge_app_name]
            
            if len(badge_apps) == 0:
                print(f"Warning: No badge application found for location: {location}")
                building_access_issues += len(location_users)
                continue
                
            badge_app_id = badge_apps.iloc[0]['application_id']
            
            # Find the access_building permission for this badge app
            building_permissions = self.permissions[
                (self.permissions['application_id'] == badge_app_id) & 
                (self.permissions['name'] == 'access_building')
            ]
            
            if len(building_permissions) == 0:
                print(f"Warning: No 'access_building' permission found for {badge_app_name}")
                building_access_issues += len(location_users)
                continue
                
            building_perm_id = building_permissions.iloc[0]['permission_id']
            
            # Check if each user has this specific permission
            for user_id in location_users['user_id']:
                user_has_access = len(self.rights[
                    (self.rights['user_id'] == user_id) &
                    (self.rights['application_id'] == badge_app_id) &
                    (self.rights['permission_id'] == building_perm_id)
                ]) > 0
                
                if not user_has_access:
                    building_access_issues += 1
                    # Debug: Uncomment to see which users are missing access
                    # print(f"User {user_id} in {location} missing building access")
        
        tests.append({
            'name': 'All users have building access in their location',
            'passed': building_access_issues == 0,
            'details': f'{building_access_issues} users missing building access'
        })
        
        # Test: Department permission consistency
        department_scores = []
        for dept in self.users['department'].unique():
            dept_users = self.users[self.users['department'] == dept]
            if len(dept_users) < 3:
                continue
                
            dept_rights = self.rights[self.rights['user_id'].isin(dept_users['user_id'])]
            perm_counts = dept_rights.groupby(['application_id', 'permission_id']).size()
            
            if len(perm_counts) > 0:
                max_consistency = perm_counts.max() / len(dept_users)
                department_scores.append(max_consistency)
        
        avg_consistency = np.mean(department_scores) if department_scores else 0
        tests.append({
            'name': f'Department permission consistency (threshold: {self.config["consistency_threshold"]})',
            'passed': avg_consistency >= self.config['consistency_threshold'],
            'details': f'Average consistency: {avg_consistency:.3f}'
        })
        
        return self._score_test_suite(tests, "Business Rules")
    
    def validate_ai_readiness(self):
        """Test suite for ML model training suitability"""
        tests = []
        
        # Test: Feature richness
        categorical_features = (
            len(self.users['department'].unique()) + 
            len(self.users['position'].unique()) + 
            len(self.users['location'].unique()) +
            len(self.users['contract_type'].unique()) +
            len(self.users['seniority'].unique())
        )
        tests.append({
            'name': 'Sufficient categorical features for ML',
            'passed': categorical_features >= 20,
            'details': f'Found {categorical_features} categorical features'
        })
        
        # Test: Class balance
        position_counts = self.users['position'].value_counts()
        min_size = position_counts.min()
        max_size = position_counts.max()
        balance_ratio = min_size / max_size if max_size > 0 else 0
        
        tests.append({
            'name': f'Position class balance (threshold: {self.config["balance_threshold"]})',
            'passed': balance_ratio >= self.config['balance_threshold'],
            'details': f'Balance ratio: {balance_ratio:.3f} (min: {min_size}, max: {max_size})'
        })
        
        # Test: Permission distribution diversity
        user_permission_counts = self.rights.groupby('user_id').size()
        if len(user_permission_counts) > 0:
            permission_entropy = user_permission_counts.value_counts(normalize=True)
            entropy_score = -np.sum(permission_entropy * np.log(permission_entropy))
        else:
            entropy_score = 0
            
        tests.append({
            'name': f'Permission distribution diversity (threshold: {self.config["entropy_threshold"]})',
            'passed': entropy_score >= self.config['entropy_threshold'],
            'details': f'Entropy score: {entropy_score:.3f}'
        })
        
        # Test: Noise level (coefficient of variation)
        if len(user_permission_counts) > 0:
            avg_rights = user_permission_counts.mean()
            std_rights = user_permission_counts.std()
            cv = std_rights / avg_rights if avg_rights > 0 else 0
        else:
            cv = 0
            
        tests.append({
            'name': f'Appropriate noise variation (CV between {self.config["cv_min"]}-{self.config["cv_max"]})',
            'passed': self.config['cv_min'] <= cv <= self.config['cv_max'],
            'details': f'Coefficient of variation: {cv:.3f}'
        })
        
        return self._score_test_suite(tests, "AI Readiness")
    
    def validate_data_patterns(self):
        """Validate realistic data patterns"""
        tests = []
        
        # Merge data for analysis
        rights_extended = self.rights.merge(
            self.users, on='user_id'
        )
        
        # Test: Seniority correlation with permissions
        if len(rights_extended) > 0:
            seniority_stats = rights_extended.groupby('seniority').agg({
                'user_id': 'nunique',
                'permission_id': 'count'
            })
            seniority_stats['perms_per_user'] = seniority_stats['permission_id'] / seniority_stats['user_id']
            
            if 'Executive' in seniority_stats.index and 'Junior' in seniority_stats.index:
                exec_perms = seniority_stats.loc['Executive', 'perms_per_user']
                junior_perms = seniority_stats.loc['Junior', 'perms_per_user']
                tests.append({
                    'name': 'Seniority-permission correlation (Executives > Juniors)',
                    'passed': exec_perms > junior_perms,
                    'details': f'Executives: {exec_perms:.1f}, Juniors: {junior_perms:.1f}'
                })
        
        # Test: Data sparsity
        total_possible = len(self.users) * len(self.permissions)
        actual_assignments = len(self.rights)
        sparsity = 1 - (actual_assignments / total_possible) if total_possible > 0 else 0
        
        tests.append({
            'name': f'Realistic data sparsity (threshold: {self.config["sparsity_threshold"]})',
            'passed': sparsity >= self.config['sparsity_threshold'],
            'details': f'Sparsity: {sparsity:.4f}'
        })
        
        return self._score_test_suite(tests, "Data Patterns")
    
    def _score_test_suite(self, tests, suite_name):
        """Calculate score for a test suite"""
        passed = sum(1 for test in tests if test['passed'])
        total = len(tests)
        score = (passed / total) * 100 if total > 0 else 0
        
        suite_results = {
            'score': score,
            'passed': passed,
            'total': total,
            'tests': tests
        }
        
        print(f"  {suite_name}: {score:.1f}% ({passed}/{total} tests passed)")
        return suite_results
    
    def _calculate_overall_score(self):
        """Calculate weighted overall score"""
        weights = {
            'validate_data_completeness': 0.25,
            'validate_business_rules': 0.20, 
            'validate_ai_readiness': 0.25,
            'validate_data_patterns': 0.15,
        }
        
        weighted_sum = 0
        for suite_name, suite_results in self.results['test_suites'].items():
            weight = weights.get(suite_name, 0.10)
            weighted_sum += suite_results['score'] * weight
        
        self.results['overall_score'] = weighted_sum
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # Analyze failures and suggest fixes
        for suite_name, suite_results in self.results['test_suites'].items():
            if suite_results['score'] < 80:
                failed_tests = [test for test in suite_results['tests'] if not test['passed']]
                for test in failed_tests[:2]:
                    recommendations.append(f"Fix: {test['name']} - {test['details']}")
        
        # General recommendations
        if self.results['overall_score'] < 80:
            recommendations.append("Consider increasing dataset size or improving business rule implementation")
        if self.results['overall_score'] >= 90:
            recommendations.append("Dataset is excellent for AI training - proceed with model development")
        
        self.results['recommendations'] = recommendations
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "=" * 60)
        print("📊 IAM DATASET VALIDATION REPORT")
        print("=" * 60)
        
        print(f"Overall Score: {self.results['overall_score']:.1f}%")
        print(f"Validation Date: {self.results['timestamp']}")
        
        print("\n📈 DETAILED RESULTS:")
        for suite_name, suite_results in self.results['test_suites'].items():
            display_name = suite_name.replace('validate_', '').replace('_', ' ').title()
            print(f"  {display_name}: {suite_results['score']:.1f}%")
        
        print("\n🔧 RECOMMENDATIONS:")
        for rec in self.results['recommendations'][:5]:
            print(f"  • {rec}")
        
        print("\n📋 KEY METRICS:")
        print(f"  • Users: {len(self.users)}")
        print(f"  • Applications: {len(self.applications)}") 
        print(f"  • Permissions: {len(self.permissions)}")
        print(f"  • Permission assignments: {len(self.rights)}")
        print(f"  • Avg permissions/user: {len(self.rights)/len(self.users):.1f}")
        
        # Final verdict
        score = self.results['overall_score']
        if score >= 90:
            print("\n🎉 EXCELLENT - Ready for AI model training!")
        elif score >= 75:
            print("\n✅ GOOD - Suitable for AI training")
        elif score >= 60:
            print("\n⚠️ FAIR - Needs improvements before AI training") 
        else:
            print("\n❌ POOR - Significant improvements required")
    
    def save_results(self, filename="validation_report.json"):
        """Save validation results to JSON file - FIXED VERSION"""
        # Create a serializable copy of results
        serializable_results = {
            'timestamp': self.results['timestamp'],
            'overall_score': float(self.results['overall_score']),
            'recommendations': self.results['recommendations'],
            'test_suites': {}
        }
        
        # Convert test suite results to serializable format
        for suite_name, suite_results in self.results['test_suites'].items():
            serializable_results['test_suites'][suite_name] = {
                'score': float(suite_results['score']),
                'passed': int(suite_results['passed']),
                'total': int(suite_results['total']),
                'tests': []
            }
            
            for test in suite_results['tests']:
                # Convert any non-serializable objects to strings
                serializable_test = {
                    'name': str(test['name']),
                    'passed': bool(test['passed']),
                    'details': str(test['details'])
                }
                serializable_results['test_suites'][suite_name]['tests'].append(serializable_test)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✓ Validation report saved to {filename}")

# Usage
if __name__ == "__main__":
    validator = IAMDatasetValidator(data_dir="iam_dataset")
    is_ready = validator.run_comprehensive_validation()
    validator.save_results()
    print(f"\nDataset AI Ready: {'YES' if is_ready else 'NO'}")