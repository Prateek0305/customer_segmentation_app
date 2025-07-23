import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🛍️ Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

model = joblib.load("kmeans_model.pkl")
df = pd.read_csv("Mall_Customers.csv")

st.sidebar.header("📂 Upload CSV File (optional)")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

gender_mapping = {"Male": 0, "Female": 1}
df['Gender'] = df['Gender'].map(gender_mapping)

X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
df['Cluster'] = model.predict(X)

cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
cluster_count = df['Cluster'].value_counts().reset_index()
cluster_count.columns = ['index', 'Cluster']

tab1, tab2 = st.tabs(["📊 Visual Insights", "📄 Cluster Summary"])

with tab1:
    st.subheader("📊 Cluster Distribution")
    fig1 = px.bar(cluster_count, x='index', y='Cluster', color='index',
                  labels={'index': 'Cluster', 'Cluster': 'Count'},
                  title='Number of Customers per Cluster',
                  color_discrete_sequence=px.colors.sequential.Plasma)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("This chart shows how many customers belong to each cluster. It helps in understanding which group is the largest or smallest.")

    st.subheader("📈 Cluster-wise Average Annual Income")
    fig2 = px.bar(cluster_summary.reset_index(), x='Cluster', y='Annual Income (k$)',
                  color='Cluster', title='Average Annual Income per Cluster',
                  color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("This shows the average income of customers in each cluster. Useful for targeting based on financial capability.")

    st.subheader("🧓 Cluster-wise Average Age")
    fig3 = px.bar(cluster_summary.reset_index(), x='Cluster', y='Age',
                  color='Cluster', title='Average Age per Cluster',
                  color_discrete_sequence=px.colors.sequential.Cividis)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("This helps visualize the age group dominating each cluster — useful for age-targeted marketing.")

    st.subheader("💸 Cluster-wise Average Spending Score")
    fig4 = px.bar(cluster_summary.reset_index(), x='Cluster', y='Spending Score (1-100)',
                  color='Cluster', title='Average Spending Score per Cluster',
                  color_discrete_sequence=px.colors.sequential.Aggrnyl)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("This chart represents how actively customers in each cluster spend. It helps prioritize high-spending groups.")

with tab2:
    st.subheader("📄 Cluster-wise Summary Table")
    st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ by Prateek</p>", unsafe_allow_html=True)
