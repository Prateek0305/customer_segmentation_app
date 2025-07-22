import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


df = pd.read_csv("Mall_Customers.csv")
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]


kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

cluster_summary = df.groupby('Cluster').mean()

st.title("🛍️ Customer Segmentation App")
st.write("Grouping customers based on purchasing behavior and demographics.")


if st.checkbox("Show Raw Data"):
    st.dataframe(df)


st.subheader("📊 Cluster Summary")
st.dataframe(cluster_summary)

st.subheader("🧭 Cluster Distribution")
cluster_counts = df['Cluster'].value_counts()
st.bar_chart(cluster_counts)


st.subheader("🧠 Income vs Spending Score (Colored by Cluster)")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', ax=ax)
st.pyplot(fig)


st.subheader("🧠 Marketing Strategy Suggestions")
st.markdown("""
- **Cluster 1**: High-income, high-spending → Target with premium offerings.
- **Cluster 2**: Young, low-income, high-spending → Focus on youth marketing & offers.
- **Cluster 3**: High income, low spenders → Re-engagement needed.
- **Cluster 4**: Professional spenders → Highlight trending and exclusive items.
- **Cluster 0**: Balanced buyers → Loyalty rewards and retention.
""")
