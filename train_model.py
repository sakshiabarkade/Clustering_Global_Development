import pandas as pd
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(
    os.path.join(DATA_DIR, "World_development_mesurement.xlsx")
)

COUNTRY_COL = "Country"

print("COLUMNS IN DATASET:")
print(df.columns.tolist())

# =========================
# SELECT FEATURES (VALID)
# =========================
features = [
    "GDP",
    "Birth Rate",
    "Infant Mortality Rate",
    "Life Expectancy Female",
    "Life Expectancy Male",
    "Internet Usage",
    "Mobile Phone Usage",
    "Health Exp % GDP",
    "Energy Usage"
]

# =========================
# CLEAN NUMERIC COLUMNS
# =========================
for col in features:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"[,$%]", "", regex=True)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# PREPROCESSING
# =========================
X = df[features]

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X = imputer.fit_transform(X)
X = scaler.fit_transform(X)

# =========================
# TRAIN MODEL
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# =========================
# EVALUATION
# =========================
sil_score = silhouette_score(X, clusters)
print("Silhouette Score:", round(sil_score, 3))

# =========================
# SAVE MODELS
# =========================
pickle.dump(kmeans, open(os.path.join(MODEL_DIR, "kmeans.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))
pickle.dump(imputer, open(os.path.join(MODEL_DIR, "imputer.pkl"), "wb"))
pickle.dump(features, open(os.path.join(MODEL_DIR, "features.pkl"), "wb"))

print("✅ Model files saved in /model folder")
