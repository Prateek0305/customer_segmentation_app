import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

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

tab1, tab2, tab3 = st.tabs(["📊 Visual Insights", "📄 Cluster Summary", "🔍 Feature Distributions"])

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

    st.subheader("📡 3D Scatter Plot of Clusters")
    fig5 = px.scatter_3d(df, x='Annual Income (k$)', y='Spending Score (1-100)', z='Age',
                         color='Cluster', title='Customer Segments in 3D',
                         color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("This 3D plot shows the relationship between income, spending, and age across clusters.")

with tab2:
    st.subheader("📄 Cluster-wise Summary Table")
    st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'), use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ by Prateek</p>", unsafe_allow_html=True)

with tab3:
    st.subheader("🎯 Distribution of Age")
    fig6, ax1 = plt.subplots()
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax1, color='#4B8BBE')
    st.pyplot(fig6)

    st.subheader("💼 Distribution of Annual Income")
    fig7, ax2 = plt.subplots()
    sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, ax=ax2, color='#306998')
    st.pyplot(fig7)

    st.subheader("🛍️ Distribution of Spending Score")
    fig8, ax3 = plt.subplots()
    sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, ax=ax3, color='#FFE873')
    st.pyplot(fig8)

    st.markdown("These histograms help analyze the spread and concentration of values for age, income, and spending score across all customers.")

