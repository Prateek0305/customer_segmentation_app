import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🎯 Customer Segmentation Dashboard")

uploaded_file = st.file_uploader("Upload a CSV file (optional)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Mall_Customers.csv")

model = joblib.load("kmeans_model.pkl")

df['Cluster'] = model.predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])

cluster_count = df['Cluster'].value_counts().reset_index()
cluster_count.columns = ['Cluster', 'Count']

fig1 = px.bar(cluster_count, x='Cluster', y='Count', color='Cluster',
              title='Number of Customers per Cluster',
              color_discrete_sequence=px.colors.qualitative.Pastel)

fig2 = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                  color='Cluster', title='Customer Segments (Income vs Spending Score)',
                  color_discrete_sequence=px.colors.qualitative.Bold,
                  hover_data=['Gender', 'Age'])

fig3 = px.box(df, x='Cluster', y='Age', color='Cluster',
              title='Age Distribution per Cluster',
              color_discrete_sequence=px.colors.qualitative.Safe)

fig4 = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                     color='Cluster', title='3D Cluster Distribution',
                     color_discrete_sequence=px.colors.qualitative.Set3)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig3, use_container_width=True)
with col4:
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("📊 Data Table with Cluster Labels")
st.dataframe(df.head(30), use_container_width=True)

st.subheader("📌 Cluster Averages")
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
st.dataframe(cluster_summary.style.background_gradient(cmap="YlGnBu"), use_container_width=True)

st.markdown("---")
st.caption("Created by Prateek Agrawal – Mall Customer Segmentation using KMeans")
