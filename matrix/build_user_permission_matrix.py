import pandas as pd
import numpy as np
from scipy import sparse
import os

INPUT_DIR = "iam_dataset"
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Chargement des données ---")
# 1. Chargement
users  = pd.read_csv(os.path.join(INPUT_DIR, "users.csv"))
apps   = pd.read_csv(os.path.join(INPUT_DIR, "applications.csv"))
perms  = pd.read_csv(os.path.join(INPUT_DIR, "permissions.csv"))
rights = pd.read_csv(os.path.join(INPUT_DIR, "rights.csv"))

# 2. Correction immédiate des noms de colonnes (Perms et Apps)
if 'name' in perms.columns and 'perm_name' not in perms.columns:
    print("Correction: Renommage de 'name' en 'perm_name' dans permissions.")
    perms = perms.rename(columns={'name': 'perm_name'})

if 'name' in apps.columns and 'app_name' not in apps.columns:
    print("Correction: Renommage de 'name' en 'app_name' dans applications.")
    apps = apps.rename(columns={'name': 'app_name'})

# 3. FILTRAGE METIER : On retire les applications 'FACILITIES'
print("--- Filtrage des données non pertinentes ---")
facility_app_ids = apps[apps['category'] == 'FACILITIES']['application_id'].tolist()
print(f"Applications exclues (Badge/Facilities) : {len(facility_app_ids)}")

# On ne garde que les permissions qui NE SONT PAS dans la liste des ID exclus
perms_metier = perms[~perms['application_id'].isin(facility_app_ids)].copy()
print(f"Permissions restantes : {len(perms_metier)} (sur {len(perms)} initiales)")

# On filtre aussi la table de liaison 'rights' pour ne garder que les liens valides
rights_metier = rights[rights['permission_id'].isin(perms_metier['permission_id'])].copy()
print(f"Droits restants : {len(rights_metier)} (sur {len(rights)} initiaux)")


print("--- Construction des Catalogues ---")
# On s'assure que les IDs sont bien triés pour que l'index de la matrice corresponde
# Utilisateurs
users_cat = users.sort_values("user_id")[["user_id", "department", "position"]].reset_index(drop=True)
users_cat["matrix_user_idx"] = users_cat.index
user_id_to_idx = dict(zip(users_cat["user_id"], users_cat["matrix_user_idx"]))

# Permissions (Métier uniquement)
perms_cat = perms_metier.sort_values("permission_id")[["permission_id", "application_id", "perm_name"]].reset_index(drop=True)
perms_cat["matrix_perm_idx"] = perms_cat.index
perm_id_to_idx = dict(zip(perms_cat["permission_id"], perms_cat["matrix_perm_idx"]))

# Applications (Pour référence)
apps_cat = apps[["application_id", "app_name", "category"]].drop_duplicates()

# Sauvegarde des catalogues
users_cat.to_csv(os.path.join(OUTPUT_DIR, "users_catalog.csv"), index=False)
perms_cat.to_csv(os.path.join(OUTPUT_DIR, "perm_catalog.csv"), index=False)
apps_cat.to_csv(os.path.join(OUTPUT_DIR, "app_catalog.csv"), index=False)


print("--- Construction de la Matrice Sparse ---")
# On map les IDs vers les index de matrice [0..N-1]
# On utilise rights_metier pour ne pas avoir d'erreur de clé
rows = rights_metier["user_id"].map(user_id_to_idx).dropna()
cols = rights_metier["permission_id"].map(perm_id_to_idx).dropna()
data = np.ones(len(rows))

n_users = len(users_cat)
n_perms = len(perms_cat)

# Matrice Creuse (Compressed Sparse Row)
M_up = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_perms))

# Sauvegarde format optimisé .npz
sparse.save_npz(os.path.join(OUTPUT_DIR, "user_permission_matrix_sparse.npz"), M_up)

print(f"Succès ! Matrice générée : {M_up.shape}")
print(f"Fichiers sauvegardés dans ./{OUTPUT_DIR}")