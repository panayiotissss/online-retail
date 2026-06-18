import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from features import log_transform

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

CLUSTER_NAMES = {0: "Champions", 1: "At Risk", 2: "Lost", 3: "Loyal but Slow"}

SEGMENT_INFO = {
    "Champions":      {"color": "#1f77b4", "emoji": "🏆", "desc": "Best customers. Buy often, spend a lot, bought recently."},
    "At Risk":        {"color": "#ff7f0e", "emoji": "⚠️",  "desc": "Used to be great customers. Haven't bought in a while — time to re-engage."},
    "Lost":           {"color": "#d62728", "emoji": "💤",  "desc": "Low engagement, long time since last purchase. Hard to recover."},
    "Loyal but Slow": {"color": "#2ca02c", "emoji": "🐢",  "desc": "Regular customers but low spend. Good upsell opportunity."},
}

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation")
st.caption("Enter a customer's RFM features to predict their segment.")


@st.cache_data
def load_umap_data():
    df = pd.read_csv("data/umap_2d.csv")
    df["Segment"] = df["cluster"].map(CLUSTER_NAMES)
    return df


@st.cache_resource
def load_artifacts():
    scaler  = joblib.load("models/scaler.pkl")
    reducer = joblib.load("models/umap_reducer.pkl")
    return scaler, reducer


# Sidebar — inputs
st.sidebar.header("Customer Features")
recency            = st.sidebar.number_input("Recency (days since last purchase)",  min_value=0,   value=30)
frequency          = st.sidebar.number_input("Frequency (number of orders)",         min_value=1,   value=5)
monetary           = st.sidebar.number_input("Monetary (total spend £)",             min_value=0.0, value=500.0)
total_items        = st.sidebar.number_input("Total Items purchased",                min_value=0,   value=50)
unique_products    = st.sidebar.number_input("Unique Products",                      min_value=1,   value=10)
tenure             = st.sidebar.number_input("Tenure (days as customer)",            min_value=0,   value=300)
avg_order_value    = st.sidebar.number_input("Avg Order Value (£)",                  min_value=0.0, value=100.0)
avg_qty_per_order  = st.sidebar.number_input("Avg Quantity per Order",               min_value=0.0, value=10.0)
avg_days_between   = st.sidebar.number_input("Avg Days Between Orders",              min_value=0.0, value=60.0)

predict_btn = st.sidebar.button("Predict Segment", use_container_width=True)

# Main area — always show the UMAP background
umap_df = load_umap_data()

col1, col2 = st.columns([2, 1])

with col1:
    if predict_btn:
        payload = {
            "Recency":              recency,
            "Frequency":            frequency,
            "Monetary":             monetary,
            "TotalItems":           total_items,
            "UniqueProducts":       unique_products,
            "Tenure":               tenure,
            "AvgOrderValue":        avg_order_value,
            "AvgQuantityPerOrder":  avg_qty_per_order,
            "AvgDaysBetweenOrders": avg_days_between,
        }

        with st.spinner("Predicting..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                result  = response.json()
                segment = result["segment"]
                info    = SEGMENT_INFO[segment]

                # Segment banner
                st.markdown(f"""
                <div style="background:{info['color']}; padding:20px; border-radius:12px; color:white; margin-bottom:20px;">
                    <h2 style="margin:0">{info['emoji']} {segment}</h2>
                    <p style="margin:6px 0 0 0; opacity:0.9">{info['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Project new point onto UMAP
                scaler, reducer = load_artifacts()
                new_df     = log_transform(pd.DataFrame([payload]))
                new_scaled = scaler.transform(new_df)
                new_2d     = reducer.transform(new_scaled)

                # Plot
                fig = px.scatter(
                    umap_df, x="x", y="y", color="Segment",
                    color_discrete_map={name: SEGMENT_INFO[name]["color"] for name in SEGMENT_INFO},
                    opacity=0.35, title="UMAP — Customer Space",
                    labels={"x": "", "y": ""},
                )
                fig.update_traces(marker=dict(size=4))

                fig.add_trace(go.Scatter(
                    x=[new_2d[0, 0]], y=[new_2d[0, 1]],
                    mode="markers",
                    marker=dict(size=18, color="black", symbol="star", line=dict(color="white", width=1)),
                    name="New Customer",
                ))

                fig.update_layout(legend_title="Segment", height=500)
                st.plotly_chart(fig, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Make sure it is running on " + API_URL)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        # Default view — just the UMAP without a new point
        fig = px.scatter(
            umap_df, x="x", y="y", color="Segment",
            color_discrete_map={name: SEGMENT_INFO[name]["color"] for name in SEGMENT_INFO},
            opacity=0.35, title="UMAP — Customer Space",
            labels={"x": "", "y": ""},
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(legend_title="Segment", height=500)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Segment Guide")
    for name, info in SEGMENT_INFO.items():
        st.markdown(f"""
        <div style="border-left: 4px solid {info['color']}; padding: 8px 12px; margin-bottom:12px; border-radius:4px; background:#f9f9f9">
            <strong>{info['emoji']} {name}</strong><br>
            <small>{info['desc']}</small>
        </div>
        """, unsafe_allow_html=True)
