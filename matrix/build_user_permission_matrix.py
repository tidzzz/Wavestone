import pandas as pd
import numpy as np
from scipy import sparse
import os

INPUT_DIR = "iam_dataset"
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- 1. Chargement des données ---")
users  = pd.read_csv(os.path.join(INPUT_DIR, "users.csv"))
apps   = pd.read_csv(os.path.join(INPUT_DIR, "applications.csv"))
perms  = pd.read_csv(os.path.join(INPUT_DIR, "permissions.csv"))
rights = pd.read_csv(os.path.join(INPUT_DIR, "rights.csv"))

# --- CORRECTIONS DE NOMS ---
if 'name' in perms.columns and 'perm_name' not in perms.columns:
    perms = perms.rename(columns={'name': 'perm_name'})
if 'name' in apps.columns and 'app_name' not in apps.columns:
    apps = apps.rename(columns={'name': 'app_name'})

# --- FILTRAGE METIER STRICT ---
# On supprime TOUT ce qui est FACILITIES. L'IA ne doit pas gérer ça.
print("--- Filtrage strict : Exclusion des Facilities ---")
facility_app_ids = apps[apps['category'] == 'FACILITIES']['application_id'].tolist()
perms_metier = perms[~perms['application_id'].isin(facility_app_ids)].copy()

# On filtre rights pour ne garder que les liens valides (sécurité de base)
rights_final = rights[rights['permission_id'].isin(perms_metier['permission_id'])].copy()
print(f"Droits conservés : {len(rights_final)}")

print("--- 2. Construction des Catalogues ---")
users_cat = users.sort_values("user_id")[["user_id", "department", "position", "location"]].reset_index(drop=True)
users_cat["matrix_user_idx"] = users_cat.index
user_id_to_idx = dict(zip(users_cat["user_id"], users_cat["matrix_user_idx"]))

perms_cat = perms_metier.sort_values("permission_id")[["permission_id", "application_id", "perm_name"]].reset_index(drop=True)
perms_cat["matrix_perm_idx"] = perms_cat.index
perm_id_to_idx = dict(zip(perms_cat["permission_id"], perms_cat["matrix_perm_idx"]))

apps_cat = apps[["application_id", "app_name", "category"]].drop_duplicates()

# Sauvegarde
users_cat.to_csv(os.path.join(OUTPUT_DIR, "users_catalog.csv"), index=False)
perms_cat.to_csv(os.path.join(OUTPUT_DIR, "perm_catalog.csv"), index=False)
apps_cat.to_csv(os.path.join(OUTPUT_DIR, "app_catalog.csv"), index=False)

print("--- 3. Construction de la Matrice ---")
rows = rights_final["user_id"].map(user_id_to_idx).dropna()
cols = rights_final["permission_id"].map(perm_id_to_idx).dropna()
data = np.ones(len(rows))

M_up = sparse.csr_matrix((data, (rows, cols)), shape=(len(users_cat), len(perms_cat)))
sparse.save_npz(os.path.join(OUTPUT_DIR, "user_permission_matrix_sparse.npz"), M_up)

print(f"Terminé ! Matrice générée : {M_up.shape}")