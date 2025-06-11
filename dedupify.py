import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import PyPDF2

# Set up Streamlit page
st.set_page_config(page_title="Text Clustering & Document Deduplication - Prerna Gyanchandani", layout="wide")

st.title("📄🧠 Text Clustering & Document Deduplication - *Prerna Gyanchandani*")
st.markdown("Upload documents or enter text manually to analyze clusters and detect duplicates.")

# --- Tabs ---
tabs = st.tabs(["📌 Text Clustering", "📂 Document Deduplication"])

# --------------
# TEXT CLUSTERING
# --------------
with tabs[0]:
    st.header("🧠 Text Clustering using Cosine Similarity")
    user_input = st.text_area("📝 Enter one document per line:", height=200)
    num_clusters = st.slider("🔢 Select number of clusters:", 2, 10, 3)

    if st.button("🚀 Cluster & Visualize"):
        documents = [doc.strip() for doc in user_input.split('\n') if doc.strip()]

        if len(documents) < 2:
            st.warning("⚠️ Please enter at least 2 documents.")
        elif num_clusters > len(documents):
            st.error(f"❌ Number of clusters ({num_clusters}) cannot exceed number of documents ({len(documents)}).")
        else:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)

            cosine_sim = cosine_similarity(tfidf_matrix)
            cosine_dist = 1 - cosine_sim

            clustering_model = AgglomerativeClustering(
                n_clusters=num_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering_model.fit_predict(cosine_dist)

            df = pd.DataFrame({'Document': documents, 'Cluster': labels})
            st.subheader("📋 Clustered Documents")
            st.dataframe(df)

            sim_df = pd.DataFrame(
                cosine_sim,
                columns=[f'Doc {i+1}' for i in range(len(documents))],
                index=[f'Doc {i+1}' for i in range(len(documents))]
            )
            st.subheader("🔵 Cosine Similarity Matrix")
            st.dataframe(sim_df.style.format("{:.2f}").background_gradient(cmap='Blues'))

            st.subheader("📊 Cluster Visualization (2D Projection)")
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(tfidf_matrix.toarray())

            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='rainbow', s=100, edgecolors='k')

            for i, (x, y) in enumerate(reduced_data):
                ax.text(x + 0.01, y + 0.01, f'Doc {i+1}', fontsize=8)

            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_title("📌 Document Clusters")
            st.pyplot(fig)

    st.markdown("""
    <div style='text-align: center;'>
        <p>Developed with ❤️ using Streamlit</p>
        <p>© 2025 Prerna Gyanchandani. All Rights Reserved.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------
# DOCUMENT DEDUPLICATION
# ------------------------
with tabs[1]:
    st.header("📂 Document Deduplication")

    def load_documents(uploaded_files):
        documents = []
        filenames = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                reader = PyPDF2.PdfReader(file)
                content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            else:
                content = file.read().decode("utf-8")
            documents.append(content)
            filenames.append(file.name)
        return filenames, documents

    def find_duplicates(filenames, documents, threshold=0.8):
        if not any(doc.strip() for doc in documents):
            st.error("All uploaded documents are empty or contain only stop words.")
            return []

        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            if tfidf_matrix.shape[1] == 0:
                st.error("No meaningful words found in the documents.")
                return []

            similarity_matrix = cosine_similarity(tfidf_matrix)
            duplicates = [
                (filenames[i], filenames[j], similarity_matrix[i, j], documents[i][:500], documents[j][:500])
                for i in range(len(filenames)) for j in range(i + 1, len(filenames))
                if similarity_matrix[i, j] > threshold
            ]
            return duplicates
        except ValueError as e:
            st.error(f"Error processing documents: {e}")
            return []

    uploaded_files = st.file_uploader("📂 Upload Documents", accept_multiple_files=True, type=["txt", "pdf"])

    if uploaded_files:
        filenames, documents = load_documents(uploaded_files)
        st.write("📄 Uploaded Files:", filenames)
        with st.spinner("🔍 Analyzing documents..."):
            duplicates = find_duplicates(filenames, documents)

        if duplicates:
            st.subheader("🔍 Duplicate Documents Found")
            for doc1, doc2, similarity, content1, content2 in duplicates:
                st.markdown(f"**📂 {doc1}** ⬌ **📂 {doc2}** (Similarity: {similarity:.2f})")
                with st.expander(f"View Content of {doc1}"):
                    st.text(content1)
                with st.expander(f"View Content of {doc2}"):
                    st.text(content2)
                st.markdown("---")
        elif any(doc.strip() for doc in documents):
            st.success("✅ No duplicates found!")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p>Developed with ❤️ using Streamlit</p>
        <p>© 2025 Prerna Gyanchandani. All Rights Reserved.</p>
    </div>
    """, unsafe_allow_html=True)
