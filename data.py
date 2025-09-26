"""
generate_iam_dataset.py
Génère 4 fichiers CSV pour le projet Role Mining / IAM :
- users.csv
- applications.csv
- permissions.csv
- rights.csv

Dépendances : pip install pandas faker numpy
"""

import random
import pandas as pd
import numpy as np
from faker import Faker
from collections import defaultdict
import os

fake = Faker('fr_FR')
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "iam_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_USERS = 5000
N_APPLICATIONS = 50            # nombre d'applications distinctes
AVG_PERMISSIONS_PER_APP = 10   # moyenne de permissions par application
TOTAL_PERMISSIONS = N_APPLICATIONS * AVG_PERMISSIONS_PER_APP

# ----- listes réalistes / customisables -----
locations = ["Paris", "Toulouse", "Nantes", "Lyon", "Marseille", "Strasbourg", "Montpellier", "Bordeaux", "Rouen"]
contract_types = ["CDI", "CDD", "Stage", "Apprentissage", "Interim"]
seniority_levels = ["Junior", "Medium", "Senior", "Lead", "Executive"]

# Positions groupés (on peut ajouter/retirer)
positions_by_group = {
    "IT": ["Sysadmin", "Dev", "DevOps", "Architect", "IT Support", "Security Engineer", "DBA"],
    "HR": ["HR Manager", "Recruiter", "Payroll Specialist"],
    "Finance": ["Accountant", "Financial Analyst", "Payroll Manager"],
    "Facilities": ["Facility Manager", "Receptionist", "Cleaner", "Security Guard"],
    "Sales": ["Sales Rep", "Account Manager", "Sales Manager"],
    "R&D": ["Researcher", "Scientist", "Engineer"],
    "Product": ["Product Manager", "Product Owner", "Business Analyst"],
    "Legal": ["Legal Counsel"],
    "Intern": ["Intern"]
}

# Flatten positions and give each a primary department name
positions = []
position_to_department = {}
for dept, pos_list in positions_by_group.items():
    for p in pos_list:
        positions.append(p)
        position_to_department[p] = f"Department {dept}"




# ====== Applications "réalistes" par catégorie ======
# Chaque appli a un name et une category, ce qui permet de définir des règles métier propres.
APPLICATION_CATALOG = [
    # Collaboration & communication
    ("Microsoft Teams", "COLLAB"),
    ("SharePoint", "COLLAB"),
    ("Exchange Online", "COLLAB"),
    ("Outlook Web", "COLLAB"),
    ("Zoom", "COLLAB"),
    ("Slack", "COLLAB"),

    # RH
    ("Workday HCM", "HR"),
    ("SAP SuccessFactors", "HR"),
    ("Talentsoft", "HR"),

    # Paie / Finance
    ("ADP Payroll", "PAYROLL"),
    ("SAP Payroll", "PAYROLL"),
    ("Sage Paie", "PAYROLL"),
    ("SAP S/4HANA Finance", "FINANCE"),
    ("Oracle Financials Cloud", "FINANCE"),

    # CRM / Sales
    ("Salesforce", "CRM"),
    ("HubSpot CRM", "CRM"),
    ("Microsoft Dynamics 365 Sales", "CRM"),

    # IT / Dev / Ops
    ("Jira", "DEVOPS"),
    ("Confluence", "DEVOPS"),
    ("GitHub Enterprise", "DEVOPS"),
    ("GitLab", "DEVOPS"),
    ("Jenkins", "DEVOPS"),
    ("ServiceNow ITSM", "ITSM"),
    ("Okta", "SECURITY"),
    ("Azure Active Directory", "SECURITY"),
    ("CrowdStrike Falcon", "SECURITY"),
    ("Nessus", "SECURITY"),

    # Data / BI
    ("Power BI", "DATA"),
    ("Tableau Server", "DATA"),
    ("Snowflake", "DATA"),
    ("Databricks", "DATA"),

    # ERP / Supply Chain
    ("SAP S/4HANA Logistics", "ERP"),
    ("Oracle NetSuite", "ERP"),

    # Bureau / Docs
    ("Google Drive", "DOCS"),
    ("Google Workspace Admin", "DOCS"),
    ("Microsoft OneDrive", "DOCS"),

    # Facilities / bâtiments (pour les accès badge)
    ("Access Badge System - HQ", "FACILITIES"),
    ("Access Badge System - Toulouse", "FACILITIES"),
    ("Access Badge System - Nantes", "FACILITIES"),
    ("Visitor Management System", "FACILITIES"),
]


# --- Générer 1 application Facilities "Access Badge System - <Ville>" par ville ---
# (On enlève d'abord d'éventuelles entrées Facilities existantes pour éviter les doublons.)
APPLICATION_CATALOG = [app for app in APPLICATION_CATALOG
                       if not (app[1] == "FACILITIES" and app[0].startswith("Access Badge System - "))]

# On ajoute une appli de badge par ville de 'locations'
facilities_per_city = [(f"Access Badge System - {city}", "FACILITIES") for city in locations]
APPLICATION_CATALOG = facilities_per_city + APPLICATION_CATALOG  # on les met en tête (optionnel)

applications = []
for i, (nm, cat) in enumerate(APPLICATION_CATALOG[:N_APPLICATIONS], start=1):
    applications.append({"application_id": i, "name": nm, "category": cat})

    
# Mapping ville -> application_id de badge (utile pour l’assignation des droits)
city_to_badge_app_id = {
    city: app["application_id"]
    for city in locations
    for app in applications
    if app["category"] == "FACILITIES" and app["name"] == f"Access Badge System - {city}"
}

# on complète par des apps génériques:
while len(APPLICATION_CATALOG) < N_APPLICATIONS:
    APPLICATION_CATALOG.append((f"Internal App {len(APPLICATION_CATALOG)+1}", "OTHER"))



apps_df = pd.DataFrame(applications)
apps_df.to_csv(os.path.join(OUTPUT_DIR, "applications.csv"), index=False)

# ----- Permissions: adapter les noms aux catégories pour + de réalisme -----
permissions = []
pid = 1
permissions_by_app = defaultdict(list)

# gabarits de permissions par catégorie (login, lecture, écriture, admin, approbation, etc.)
PERM_TEMPLATES = {
    "COLLAB":      ["login", "create_channel", "post_message", "read_messages", "manage_team"],
    "HR":          ["login", "view_employee", "edit_employee", "export_hr_reports", "manage_recruiting"],
    "PAYROLL":     ["login", "view_payslips", "submit_payroll", "approve_payroll", "admin_payroll"],
    "FINANCE":     ["login", "view_invoices", "post_journal", "approve_payment", "admin_finance"],
    "CRM":         ["login", "view_accounts", "edit_opportunities", "export_leads", "admin_crm"],
    "DEVOPS":      ["login", "create_issue", "merge_request", "manage_pipeline", "admin_project"],
    "ITSM":        ["login", "open_ticket", "resolve_ticket", "view_cmdb", "admin_itsm"],
    "SECURITY":    ["login", "view_alerts", "quarantine_endpoint", "manage_identities", "admin_security"],
    "DATA":        ["login", "view_dashboard", "publish_report", "query_warehouse", "admin_data"],
    "ERP":         ["login", "view_orders", "post_goods_movement", "approve_po", "admin_erp"],
    "DOCS":        ["login", "read_document", "edit_document", "share_document", "admin_docs"],
    "FACILITIES":  ["login", "access_building", "request_badge", "manage_visitors", "admin_facilities"],
    "OTHER":       ["login", "read", "write", "export", "admin"],
}

permissions = []
pid = 1
permissions_by_app = defaultdict(list)

# Tu dois avoir PERM_TEMPLATES défini (comme dans ta version "apps réelles")
# On garantit la présence de 'access_building' pour TOUTE app FACILITIES
access_building_pid_by_app = {}  # app_id -> permission_id 'access_building'

for app in applications:
    app_id = app["application_id"]
    cat = app["category"]
    base = PERM_TEMPLATES.get(cat, PERM_TEMPLATES["OTHER"])

    # nb de permissions aléatoires + gabarits
    n_perm = max(5, int(np.random.poisson(9)))
    names = set()
    while len(names) < n_perm:
        action = random.choice(base)
        target = random.choice(["", "_users", "_data", "_reports", "_project", "_settings", "_building", "_mailbox"])
        perm_name = f"{action}{target}".strip("_")
        names.add(perm_name)

    # >>> Garantie 'access_building' pour les apps de badge
    if cat == "FACILITIES":
        names.add("access_building")

    # Création des permissions + index inverses
    for name in sorted(names):
        permissions.append({
            "application_id": app_id,
            "permission_id": pid,
            "name": name
        })
        permissions_by_app[app_id].append(pid)

        # Mémoriser l'ID de 'access_building'
        if app["category"] == "FACILITIES" and name == "access_building":
            access_building_pid_by_app[app_id] = pid

        pid += 1

perms_df = pd.DataFrame(permissions)
perms_df.to_csv(os.path.join(OUTPUT_DIR, "permissions.csv"), index=False)


# ----- Sélections par catégorie (remplace les filtres "if 'Collab' in name") -----
def apps_by_category(cat):
    return [a["application_id"] for a in applications if a["category"] == cat]

apps_it         = apps_by_category("DEVOPS") + apps_by_category("ITSM") + apps_by_category("SECURITY") + apps_by_category("DATA")
apps_hr         = apps_by_category("HR")
apps_payroll    = apps_by_category("PAYROLL") + apps_by_category("FINANCE")
apps_facilities = apps_by_category("FACILITIES")
collab_apps     = apps_by_category("COLLAB")

# ----- règles métier de base -----
# mapping "base permissions" par grand rôle/position (application_id & permission_id)
# on va définir pour chaque "position group" des permissions types (par app).
base_rules = defaultdict(list)



# For each group, select some typical permissions (by choosing random perms in certain apps)
def pick_perms(app_list, k=3):
    chosen = []
    if not app_list:
        return chosen
    for _ in range(k):
        app = random.choice(app_list)
        perm = random.choice(permissions_by_app[app])
        chosen.append((app, perm))
    return chosen

# Rules for groups (coarse)
base_rules["IT"] = pick_perms(apps_it, k=6) + pick_perms(apps_hr, k=1)
base_rules["HR"] = pick_perms(apps_hr, k=5) + pick_perms(apps_payroll, k=2)
base_rules["Finance"] = pick_perms(apps_payroll, k=5) + pick_perms(apps_it, k=1)
base_rules["Facilities"] = pick_perms(apps_facilities, k=4)
base_rules["Sales"] = pick_perms(apps_it, k=2) + pick_perms(apps_facilities, k=1)
base_rules["R&D"] = pick_perms(apps_it, k=3)
base_rules["Product"] = pick_perms(apps_it, k=3)
base_rules["Legal"] = pick_perms(apps_it, k=2)
base_rules["Intern"] = pick_perms(apps_it, k=2)

# Location-based rule: employees in Toulouse have access to a Toulouse building badge app if exists
# We'll implement as: if a permission name includes "building" (access_building) we assign by location
# Find all permission ids whose name contains "building"
building_perms = []
for p in permissions:
    if "building" in p["name"]:
        building_perms.append((p["application_id"], p["permission_id"]))

# ----- Génération Users -----
users = []
for uid in range(1, N_USERS + 1):
    first = fake.first_name()
    last = fake.last_name()
    # sample position: weighted so IT/Business roles common, executives rare
    pos = random.choices(positions, weights=[5 if "Intern" in p else 20 for p in positions])[0]
    dept = position_to_department[pos]
    location = random.choice(locations)
    contract = random.choices(contract_types, weights=[50, 20, 15, 10, 5])[0]
    seniority = random.choices(seniority_levels, weights=[50, 30, 15, 4, 1])[0]
    users.append({
        "user_id": uid,
        "first_name": first,
        "last_name": last,
        "position": pos,
        "department": dept,
        "location": location,
        "contract_type": contract,
        "seniority": seniority
    })

users_df = pd.DataFrame(users)
users_df.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)

# ----- Génération Rights (habilitation) -----
rights = []

PROB_MISSING_BASE_PERMISSION = 0.02
PROB_EXTRA_PERMISSION = 0.03

all_perm_ids = [p["permission_id"] for p in permissions]
perm_to_app = {p["permission_id"]: p["application_id"] for p in permissions}

for u in users:
    uid = u["user_id"]
    pos = u["position"]
    user_city = u["location"]

    # ----- règles de base par groupe -----
    group = None
    for g, pos_list in positions_by_group.items():
        if pos in pos_list:
            group = g
            break
    base_perms = list(base_rules.get(group, []))  # [(app_id, perm_id), ...]

    # (Ancienne logique aléatoire "building_perms" → À SUPPRIMER)
    # if u["location"] in ["Toulouse", "Paris", "Nantes"]:
    #     if building_perms:
    #         bp = random.choice(building_perms)
    #         base_perms.append(bp)

    # Dédup avant bruit
    base_perms = list(set(base_perms))

    # ----- bruit : retirer quelques perms de base -----
    final_perms = []
    for (app_id, perm_id) in base_perms:
        if random.random() < PROB_MISSING_BASE_PERMISSION:
            continue
        final_perms.append((uid, app_id, perm_id))

    # ----- bruit : permissions en trop -----
    if random.random() < PROB_EXTRA_PERMISSION:
        n_extra = random.randint(1, 5)
        existing_perm_ids = set(p for (_, _, p) in final_perms)
        candidates = [p for p in all_perm_ids if p not in existing_perm_ids]
        for _ in range(n_extra):
            if not candidates:
                break
            extra = random.choice(candidates)
            candidates.remove(extra)
            final_perms.append((uid, perm_to_app[extra], extra))

    # ----- baseline collab (facultatif, comme avant) -----
    if collab_apps:
        for _ in range(random.randint(1, 2)):
            appc = random.choice(collab_apps)
            permc = random.choice(permissions_by_app[appc])
            if (uid, appc, permc) not in final_perms:
                final_perms.append((uid, appc, permc))

    # ====== OBLIGATOIRE : accès bâtiment de SA VILLE ======
    # On récupère l'app_id de badge de la ville, puis l'ID de permission 'access_building'.
    badge_app_id = city_to_badge_app_id[user_city]
    building_pid = access_building_pid_by_app[badge_app_id]
    if (uid, badge_app_id, building_pid) not in final_perms:
        final_perms.append((uid, badge_app_id, building_pid))

    rights.extend(final_perms)

rights_df = pd.DataFrame(rights, columns=["user_id", "application_id", "permission_id"])
rights_df.to_csv(os.path.join(OUTPUT_DIR, "rights.csv"), index=False)


# ----- Save applications.csv and permissions.csv -----
apps_df = pd.DataFrame(applications)
apps_df.to_csv(os.path.join(OUTPUT_DIR, "applications.csv"), index=False)

perms_df = pd.DataFrame(permissions)
perms_df.to_csv(os.path.join(OUTPUT_DIR, "permissions.csv"), index=False)

print("Fichiers générés dans:", OUTPUT_DIR)
print("Nombre users:", len(users_df))
print("Nombre droits (rows):", len(rights_df))
print("Nombre applications:", len(apps_df))
print("Nombre permissions:", len(perms_df))
