import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Global Development Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { background-color: #f9fbfd; }
h1, h2, h3 { color: #1f4e79; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Global Development Categorization Dashboard")
st.caption("Interactive ML dashboard for global socio-economic analysis")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

COUNTRY_COL = "Country"

# =========================
# LOAD MODELS & FEATURES
# =========================
scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
imputer = pickle.load(open(os.path.join(MODEL_DIR, "imputer.pkl"), "rb"))
features = pickle.load(open(os.path.join(MODEL_DIR, "features.pkl"), "rb"))

# =========================
# LOAD & CLEAN DATA
# =========================
df = pd.read_excel(os.path.join(DATA_DIR, "World_development_mesurement.xlsx"))

for col in features:
    df[col] = (
        df[col].astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", np.nan)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features).reset_index(drop=True)

X = imputer.transform(df[features])
X = scaler.transform(X)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("🎛 Dashboard Controls")

# 🔁 LIVE K-SLIDER
k = st.sidebar.slider("🔁 Number of Clusters (k)", 2, 6, 3)

selected_country = st.sidebar.selectbox(
    "🌍 Select Country",
    sorted(df[COUNTRY_COL].unique())
)

selected_features = st.sidebar.multiselect(
    "📊 Select Indicators",
    features,
    default=features[:4]
)

# =========================
# LIVE KMEANS TRAINING
# =========================
kmeans_live = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans_live.fit_predict(X)

# =========================
# CLUSTER LABELING (GDP-BASED)
# =========================
cluster_means = df.groupby("Cluster")["GDP"].mean()
cluster_order = cluster_means.sort_values().index

label_pool = [
    "Under-Developed",
    "Developing",
    "Developed",
    "High Income",
    "Very High Income",
    "Elite"
]

cluster_map = dict(zip(cluster_order, label_pool[:k]))
df["Development Level"] = df["Cluster"].map(cluster_map)

# =========================
# METRICS
# =========================
sil_score = round(silhouette_score(X, df["Cluster"]), 3)

k1, k2, k3 = st.columns(3)
k1.metric("🌍 Countries", len(df))
k2.metric("🟢 Developed %",
          round((df["Development Level"] == "Developed").mean() * 100, 2))
k3.metric("📈 Silhouette Score", sil_score)

st.divider()

# =========================
# 🌍 WORLD MAP
# =========================
st.subheader("🌍 Global Development Map")

world_map = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="Development Level",
    hover_name="Country",
    color_discrete_sequence=px.colors.qualitative.Set2
)

st.plotly_chart(world_map, use_container_width=True)

st.divider()

# =========================
# 🌍 COUNTRY DRILLDOWN MAP (NEW)
# =========================
st.subheader("📍 Country-Level Drilldown")

drill_df = df[df[COUNTRY_COL] == selected_country]

drill_map = px.choropleth(
    drill_df,
    locations="Country",
    locationmode="country names",
    color="Development Level",
    hover_name="Country",
    scope="world"
)

st.plotly_chart(drill_map, use_container_width=True)

st.divider()

# =========================
# CATEGORY TABLES
# =========================
st.subheader("📊 Country Categorization")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 🔴 Under-Developed")
    st.dataframe(df[df["Development Level"] == "Under-Developed"][[COUNTRY_COL]], height=260)

with c2:
    st.markdown("### 🟡 Developing")
    st.dataframe(df[df["Development Level"] == "Developing"][[COUNTRY_COL]], height=260)

with c3:
    st.markdown("### 🟢 Developed")
    st.dataframe(df[df["Development Level"] == "Developed"][[COUNTRY_COL]], height=260)

st.divider()

# =========================
# PIE CHART
# =========================
st.subheader("📈 Development Category Distribution")

category_counts = df["Development Level"].value_counts().reset_index()
category_counts.columns = ["Category", "Count"]

pie_fig = px.pie(
    category_counts,
    names="Category",
    values="Count",
    hole=0.4
)

st.plotly_chart(pie_fig, use_container_width=True)

st.divider()

# =========================
# RADAR CHART
# =========================
st.subheader("📌 Country vs Cluster Average")

row = df[df[COUNTRY_COL] == selected_country].iloc[0]
cluster_avg = df[df["Cluster"] == row["Cluster"]][features].mean()

radar_df = pd.DataFrame({
    "Indicator": selected_features,
    "Country": [row[f] for f in selected_features],
    "Cluster Average": [cluster_avg[f] for f in selected_features]
})

radar_fig = px.line_polar(
    radar_df.melt(id_vars="Indicator"),
    r="value",
    theta="Indicator",
    color="variable",
    line_close=True
)

st.plotly_chart(radar_fig, use_container_width=True)

st.divider()

# =========================
# DISTRIBUTION PLOT
# =========================
st.subheader("📊 Indicator Distribution")

dist_feature = st.selectbox("Select Indicator", features)

box_fig = px.box(
    df,
    y=dist_feature,
    color="Development Level",
    points="all"
)

st.plotly_chart(box_fig, use_container_width=True)

st.divider()

# =========================
# SINGLE COUNTRY ANALYSIS
# =========================
st.subheader("📌 Single Country Analysis")

global_mean = df[features].mean()

def improvement_label(value, mean):
    if value < mean * 0.8:
        return "Needs Improvement"
    elif value > mean * 1.2:
        return "Good"
    else:
        return "Average"

analysis_df = pd.DataFrame([
    {
        "Indicator": f,
        "Value": round(row[f], 2),
        "Status": improvement_label(row[f], global_mean[f])
    } for f in features
])

st.dataframe(analysis_df)

# =========================
# DOWNLOAD REPORT
# =========================
st.download_button(
    "⬇️ Download Country Report (CSV)",
    data=analysis_df.to_csv(index=False),
    file_name=f"{selected_country}_development_report.csv"
)

st.success("✅ Interactive clustering & exploration complete")
