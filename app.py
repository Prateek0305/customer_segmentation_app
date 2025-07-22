import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import joblib

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🎯 Customer Segmentation Dashboard</h1>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload your customer data CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")
    st.dataframe(data.head())

    model = joblib.load("kmeans_model.pkl")
    data['Cluster'] = model.predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])

    cluster_count = data['Cluster'].value_counts().sort_index()
    cluster_count_df = cluster_count.reset_index()
    cluster_count_df.columns = ['Cluster', 'Count']

    fig1 = px.bar(cluster_count_df, x='Cluster', y='Count', color='Cluster',
                  title='Number of Customers per Cluster',
                  color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(data, x='Annual Income (k$)', y='Spending Score (1-100)',
                      color='Cluster', symbol='Cluster',
                      title='Income vs Spending Score by Cluster',
                      color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(data, x='Cluster', y='Age', color='Cluster',
                  title='Age Distribution per Cluster',
                  color_discrete_sequence=px.colors.qualitative.Prism)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Cluster-wise Averages"):
        cluster_summary = data.groupby('Cluster').mean(numeric_only=True)
        st.dataframe(cluster_summary.style.background_gradient(cmap='YlGnBu'))
