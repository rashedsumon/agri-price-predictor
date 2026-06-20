import streamlit as st
import pandas as pd
import datetime
import plotly.express as px

from data_loader import load_raw_data
from model import preprocess_and_train, predict_future_price

# Page Configuration Setup
st.set_page_config(page_title="Agri-Price AI Forecaster", page_icon="🌾", layout="wide")

st.title("🌾 Agricultural Produce Price Forecast Dashboard")
st.write("Predicting vegetable and fruit prices utilizing historical time-series datasets.")
st.markdown("---")

# Step 1: Cache and load data automatically
@st.cache_data
def get_cached_data():
    return load_raw_data()

try:
    with st.spinner("Fetching data from Kaggle... Please wait..."):
        raw_df = get_cached_data()
except Exception as e:
    st.error(f"Failed to access dataset files: {e}")
    st.stop()

# Step 2: Cache Model training processes 
@st.cache_resource
def get_trained_pipeline(_df):
    return preprocess_and_train(_df)

with st.spinner("Training predictive machine learning core..."):
    model, commodity_mapping, cleaned_df = get_trained_pipeline(raw_df)

# Main Page UI Configurations (Replaces the old Sidebar setup)
st.subheader("🔮 Price Prediction Setup")

# Split settings into columns for a neat horizontal layout
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    available_commodities = sorted(list(commodity_mapping.keys()))
    default_index = available_commodities.index("Tomato Small(Local)") if "Tomato Small(Local)" in available_commodities else 0
    selected_item = st.selectbox("Select Commodity Item:", available_commodities, index=default_index)

with col2:
    selected_date = st.date_input("Target Prediction Date:", datetime.date(2026, 6, 20))

with col3:
    # Adding vertical alignment spacing to clean up layout presentation
    st.write("##") 
    run_prediction = st.button("Run AI Prediction", use_container_width=True)

# Execution Action Trigger Logic
if run_prediction:
    predicted_price = predict_future_price(model, commodity_mapping, selected_item, selected_date)
    
    if predicted_price:
        st.success(f"### Predicted Average Price for **{selected_item}**: ₨. **{predicted_price:.2f}** / Kg")
    else:
        st.error("Item mapping calculation error.")

st.markdown("---")

# Historical Trend Data Chart Visualization
st.subheader(f"📈 Historical Price Trends for {selected_item}")
item_history = cleaned_df[cleaned_df['Commodity'] == selected_item].sort_values(by='Date')

if not item_history.empty:
    fig = px.line(item_history, x='Date', y='Average', title=f"Price history metric layout (Rs per KG) for {selected_item}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No matching item history timeline records found to plot.")
