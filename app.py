import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Segmentation Web App")
st.markdown("Group customers based on **demographics** and **purchasing behavior** to enhance marketing strategies.")

st.sidebar.header("📁 Upload Customer Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Mall_Customers.csv") 

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
cluster_summary = df.groupby('Cluster').mean().reset_index()

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Insights", "🧠 Strategy"])

with tab1:
    st.subheader("📊 Clustered Customer Data")
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

    st.subheader("📌 Cluster Distribution")
    cluster_count = df['Cluster'].value_counts().reset_index()
   cluster_count_df = cluster_count.reset_index()
cluster_count_df.columns = ['Cluster', 'Count']

fig1 = px.bar(cluster_count_df, x='Cluster', y='Count', color='Cluster',
              title='Number of Customers per Cluster')

    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("💡 Cluster Summary (Means)")
    st.dataframe(cluster_summary.style.background_gradient(cmap='YlGnBu'), use_container_width=True)

    st.subheader("💰 Income vs Spending Score (colored by cluster)")
    fig2 = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                      color='Cluster', hover_data=['Age'],
                      color_continuous_scale='rainbow',
                      title="Income vs Spending")
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧓 Age Distribution by Cluster")
        fig3 = px.box(df, x='Cluster', y='Age', color='Cluster')
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("👰🏼‍♀️ Gender Distribution")
        gender_counts = df['Gender'].value_counts().rename({0: 'Male', 1: 'Female'}).reset_index()
        fig4 = px.pie(gender_counts, values='Gender', names='index', title='Gender Ratio')
        st.plotly_chart(fig4, use_container_width=True)


with tab3:
    st.subheader("🎯 Cluster-Based Marketing Strategies")
    st.markdown("""
    - **Cluster 0**: Balanced buyers → Focus on loyalty rewards & retention
    - **Cluster 1**: High spenders with high income → Target with premium services & exclusive offers
    - **Cluster 2**: Young, high spending, low income → Engage with trend-based, budget-friendly products
    - **Cluster 3**: High income, low spenders → Use reactivation & trust-building campaigns
    - **Cluster 4**: Middle-income impulsive buyers → Push with flash sales & new arrivals
    """)

    st.markdown("---")
    st.markdown("✅ Built with `Streamlit`, `Plotly`, and `scikit-learn`")


st.markdown(
    "<hr style='margin-top: 3rem;'><center>© 2025 Prateek Agrawal | Customer Segmentation App</center>",
    unsafe_allow_html=True
)
