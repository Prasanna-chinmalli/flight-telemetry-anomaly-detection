import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Flight Telemetry Anomaly Detection System",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("✈️ Flight Telemetry Anomaly Detection System")

st.write(
    """
    A Python + Streamlit aerospace analytics dashboard
    using Isolation Forest to detect anomalies in
    simulated aircraft telemetry data.
    """
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Detection Settings")

contamination = st.sidebar.slider(
    "Anomaly Sensitivity",
    min_value=0.01,
    max_value=0.10,
    value=0.03,
    step=0.01
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("flight_data.csv")

df["time"] = pd.to_datetime(df["time"])

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------

features = [
    "altitude",
    "speed",
    "engine_temperature",
    "fuel_flow",
    "vibration"
]

# ---------------------------------------------------
# ISOLATION FOREST MODEL
# ---------------------------------------------------

model = IsolationForest(
    contamination=contamination,
    random_state=42
)

df["anomaly"] = model.fit_predict(df[features])

df["anomaly_label"] = df["anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

# ---------------------------------------------------
# DATASET VIEW
# ---------------------------------------------------

st.subheader("Flight Telemetry Dataset")

st.dataframe(df.head())

# ---------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------

normal_count = (df["anomaly_label"] == "Normal").sum()

anomaly_count = (df["anomaly_label"] == "Anomaly").sum()

health_score = round(
    (normal_count / len(df)) * 100,
    2
)

st.subheader("Anomaly Detection Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Normal Points", normal_count)

col2.metric("Anomalies Detected", anomaly_count)

col3.metric(
    "Aircraft Health Score",
    f"{health_score}%"
)

# ---------------------------------------------------
# HEALTH BAR
# ---------------------------------------------------

st.progress(int(health_score))

# ---------------------------------------------------
# ALERTS
# ---------------------------------------------------

if anomaly_count > 20:
    st.error(
        "⚠️ Warning: High anomaly activity detected!"
    )
else:
    st.success(
        "✅ Flight telemetry operating normally."
    )

# ---------------------------------------------------
# LATEST TELEMETRY
# ---------------------------------------------------

st.subheader("Latest Telemetry Reading")

st.dataframe(df.tail(1))

# ---------------------------------------------------
# TELEMETRY VISUALIZATION
# ---------------------------------------------------

st.subheader("Telemetry Visualization")

selected_feature = st.selectbox(
    "Select telemetry parameter",
    features
)

fig = px.scatter(
    df,
    x="time",
    y=selected_feature,
    color="anomaly_label",
    title=f"{selected_feature} Over Time",
    color_discrete_map={
        "Normal": "blue",
        "Anomaly": "red"
    }
)

st.plotly_chart(
    fig,
    width='stretch'
)

# ---------------------------------------------------
# FEATURE RELATIONSHIP VIEW
# ---------------------------------------------------

st.subheader("Feature Relationship View")

x_axis = st.selectbox(
    "Select X-axis",
    features,
    index=0
)

y_axis = st.selectbox(
    "Select Y-axis",
    features,
    index=2
)

fig2 = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="anomaly_label",
    title=f"{x_axis} vs {y_axis}",
    color_discrete_map={
        "Normal": "blue",
        "Anomaly": "red"
    }
)

st.plotly_chart(
    fig2,
    width='stretch'
)

# ---------------------------------------------------
# HEATMAP
# ---------------------------------------------------

st.subheader("Telemetry Correlation Heatmap")

corr = df[features].corr()

fig3, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig3)

# ---------------------------------------------------
# ANOMALY TABLE
# ---------------------------------------------------

st.subheader("Detected Anomalous Records")

anomalies_df = df[
    df["anomaly_label"] == "Anomaly"
]

st.dataframe(anomalies_df)