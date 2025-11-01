# -*- coding: utf-8 -*-
"""
Construit les matrices binaires:
  - user × permission (0/1)
  - user × application (0/1, déduite de user×permission via permission→app)

Entrées attendues (CSV en UTF-8) dans ./data :
  users.csv:        user_id, ...                      (au minimum user_id)
  applications.csv: application_id, app_name
  permissions.csv:  permission_id, application_id, perm_name
  rights.csv:       user_id, permission_id                  (optionnellement application_id)

Sorties (dans ./out) :
  - user_permission_matrix.csv
  - user_permission_matrix_sparse.npz
  - user_application_matrix.csv
  - user_application_matrix_sparse.npz
  - perm_catalog.csv
  - app_catalog.csv
  - users_catalog.csv
"""

import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz

INPUT_DIR  = "./iam_dataset"
OUTPUT_DIR = "./out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Chargement ----------
users = pd.read_csv(os.path.join(INPUT_DIR, "users.csv"))
apps  = pd.read_csv(os.path.join(INPUT_DIR, "applications.csv"))
perms = pd.read_csv(os.path.join(INPUT_DIR, "permissions.csv"))
rights = pd.read_csv(os.path.join(INPUT_DIR, "rights.csv"))

# ---------- Sanity checks légers (non bloquants) ----------
def ensure_col(df, col, dfname):
    if col not in df.columns:
        raise ValueError(f"Colonne manquante: '{col}' dans {dfname}")

ensure_col(users, "user_id", "users")
ensure_col(apps, "application_id", "applications")
# On utilise la colonne "name" et on la renomme en "app_name" pour la cohérence
if "name" in apps.columns and "app_name" not in apps.columns:
    apps = apps.rename(columns={"name": "app_name"})

ensure_col(apps, "app_name", "applications") # On s'assure que la colonne existe bien


ensure_col(perms, "permission_id", "permissions")
ensure_col(perms, "application_id", "permissions")
if "perm_name" not in perms.columns:
    perms["perm_name"] = perms["permission_id"]

ensure_col(rights, "user_id", "rights")
# rights doit relier à des permissions; si application_id est présent on l’ignore (on repart de permission_id)
if "permission_id" not in rights.columns:
    raise ValueError("rights.csv doit contenir au minimum: user_id, permission_id")

# ---------- Catalogues propres ----------
users_cat = users[["user_id"]].drop_duplicates().reset_index(drop=True)
apps_cat  = apps[["application_id", "app_name"]].drop_duplicates().reset_index(drop=True)
perms_cat = perms[["permission_id", "application_id", "name"]].drop_duplicates().reset_index(drop=True)

# Sauvegarde catalogues (utile pour tracer les colonnes de matrices)
users_cat.to_csv(os.path.join(OUTPUT_DIR, "users_catalog.csv"), index=False)
apps_cat.to_csv(os.path.join(OUTPUT_DIR, "app_catalog.csv"), index=False)
perms_cat.to_csv(os.path.join(OUTPUT_DIR, "perm_catalog.csv"), index=False)

# ---------- Liens user → permission (nettoyés sur le catalogue) ----------
rights_up = rights[["user_id", "permission_id"]].drop_duplicates()
rights_up = rights_up.merge(users_cat, on="user_id", how="inner")
rights_up = rights_up.merge(perms_cat[["permission_id"]], on="permission_id", how="inner")

# ---------- Index stables ----------
users_cat = users_cat.reset_index(drop=True)
perms_cat = perms_cat.reset_index(drop=True)
apps_cat  = apps_cat.reset_index(drop=True)

users_cat["row"] = np.arange(len(users_cat))
perms_cat["col"] = np.arange(len(perms_cat))
apps_cat["col"]  = np.arange(len(apps_cat))

# ---------- Matrice user × permission ----------
UP = rights_up.merge(users_cat[["user_id", "row"]], on="user_id", how="inner") \
              .merge(perms_cat[["permission_id", "col"]], on="permission_id", how="inner")

rows = UP["row"].to_numpy()
cols = UP["col"].to_numpy()
data = np.ones(len(UP), dtype=np.uint8)
M_up = csr_matrix((data, (rows, cols)), shape=(len(users_cat), len(perms_cat)))

# Exports (dense + sparse)
dense_up = pd.DataFrame(M_up.toarray(), columns=perms_cat.sort_values("col")["permission_id"])
dense_up.insert(0, "user_id", users_cat.sort_values("row")["user_id"].values)
dense_up.to_csv(os.path.join(OUTPUT_DIR, "user_permission_matrix.csv"), index=False)
save_npz(os.path.join(OUTPUT_DIR, "user_permission_matrix_sparse.npz"), M_up)

# ---------- Matrice user × application ----------
# Déduite via permission → app
perm_to_app = perms_cat[["permission_id", "application_id"]]
UA_links = rights_up.merge(perm_to_app, on="permission_id", how="inner")[["user_id", "application_id"]].drop_duplicates()

UA = UA_links.merge(users_cat[["user_id", "row"]], on="user_id", how="inner") \
             .merge(apps_cat[["application_id", "col"]], on="application_id", how="inner")

rows = UA["row"].to_numpy()
cols = UA["col"].to_numpy()
data = np.ones(len(UA), dtype=np.uint8)
M_ua = csr_matrix((data, (rows, cols)), shape=(len(users_cat), len(apps_cat)))

dense_ua = pd.DataFrame(M_ua.toarray(), columns=apps_cat.sort_values("col")["application_id"])
dense_ua.insert(0, "user_id", users_cat.sort_values("row")["user_id"].values)
dense_ua.to_csv(os.path.join(OUTPUT_DIR, "user_application_matrix.csv"), index=False)
save_npz(os.path.join(OUTPUT_DIR, "user_application_matrix_sparse.npz"), M_ua)

print("✅ Terminé")
print(f"- {OUTPUT_DIR}/user_permission_matrix.csv (+ .npz)")
print(f"- {OUTPUT_DIR}/user_application_matrix.csv (+ .npz)")
print(f"- {OUTPUT_DIR}/perm_catalog.csv, app_catalog.csv, users_catalog.csv")
