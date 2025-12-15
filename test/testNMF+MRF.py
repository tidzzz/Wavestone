import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

# --- 1. Chargement et Préparation (Comme avant) ---
users_df = pd.read_csv('./iam_dataset/users.csv')
rights_df = pd.read_csv('./iam_dataset/rights.csv')

# Matrice User x Permissions
user_perm_matrix = rights_df.pivot_table(index='user_id', columns='permission_id', aggfunc='size', fill_value=0)
user_perm_matrix = user_perm_matrix.reindex(users_df['user_id'], fill_value=0)

# --- 2. Étape NMF (Définition des Rôles) ---
n_roles = 5
nmf = NMF(n_components=n_roles, init='random', random_state=42, max_iter=500)
W = nmf.fit_transform(user_perm_matrix) # Poids User-Rôle
H = nmf.components_               # Poids Rôle-Permission

# On définit les droits de chaque rôle via un seuil (Thresholding)
# Un rôle contient une permission si le poids dans H est significatif
threshold = 0.01  # À ajuster selon vos données
role_definitions = {} 
for role_id in range(n_roles):
    # Quels droits font partie du rôle X ?
    role_perms = np.where(H[role_id] > threshold)[0] 
    # (Note: ce sont les indices colonnes, il faudrait mapper vers permission_id réel si besoin)
    role_definitions[role_id] = set(role_perms)

# Labeling : Rôle dominant pour chaque user
users_df['role_label'] = np.argmax(W, axis=1)

# --- 3. Étape Random Forest (Prédiction) ---
features_cols = ['position', 'department', 'location', 'contract_type', 'seniority']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X = encoder.fit_transform(users_df[features_cols])
y = users_df['role_label']

# Split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, users_df.index, test_size=0.2, random_state=42
)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# --- 4. CALCUL DES MÉTRIQUES MÉTIER (Le cœur de votre demande) ---

over_provisioning_scores = []
under_provisioning_scores = []

# Pour chaque utilisateur du test set
for i, user_idx in enumerate(idx_test):
    # 1. La Vérité Terrain (Droits réels)
    # On récupère la ligne correspondante dans la matrice originale
    # indices des droits réels (où valeur > 0)
    real_rights = set(np.where(user_perm_matrix.iloc[user_idx] > 0)[0])
    
    # 2. La Prédiction (Droits donnés par le Rôle prédit)
    predicted_role = y_pred[i]
    granted_rights = role_definitions[predicted_role]
    
    # 3. Comparaison (Ensemble)
    if len(granted_rights) > 0:
        # Droits donnés en trop (Intersection vide / Total donnés)
        excess = granted_rights - real_rights
        over_prov = len(excess) / len(granted_rights)
    else:
        over_prov = 0.0
        
    if len(real_rights) > 0:
        # Droits manqués (Intersection vide / Total réels)
        missing = real_rights - granted_rights
        under_prov = len(missing) / len(real_rights)
    else:
        under_prov = 0.0
        
    over_provisioning_scores.append(over_prov)
    under_provisioning_scores.append(under_prov)

# --- 5. RAPPORT FINAL ---

print("=== RAPPORT DE PERFORMANCE (NMF + Random Forest) ===")
print(f"1. Performance Pure (F1-Score Macro) : {f1_score(y_test, y_pred, average='macro'):.2%}")
print(f"   (Capacité à retrouver le bon cluster théorique)")
print("-" * 30)
print(f"2. RISQUE SÉCURITÉ (Sur-attribution moyenne) : {np.mean(over_provisioning_scores):.2%}")
print(f"   Interpretation : En moyenne, {np.mean(over_provisioning_scores):.1%} des droits donnés sont inutiles.")
print("-" * 30)
print(f"3. RISQUE MÉTIER (Sous-attribution moyenne)  : {np.mean(under_provisioning_scores):.2%}")
print(f"   Interpretation : En moyenne, l'utilisateur perd {np.mean(under_provisioning_scores):.1%} de ses accès nécessaires.")