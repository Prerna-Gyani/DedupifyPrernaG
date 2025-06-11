
# 📄 Dedupify

Developed by [Prerna Gyanchandani]

Dedupify is a user-friendly Streamlit web application for performing intelligent text clustering and document deduplication. It allows users to either input free-form text or upload .txt/.pdf files to analyze similarities, cluster texts, and identify duplicate content using machine learning techniques.

🚀 Features 

🔹 Text Clustering

Group multiple user-entered text documents into clusters using TF-IDF and cosine similarity

Visualize clusters in 2D using PCA (Principal Component Analysis)

Explore similarity matrix between text entries

🔹 Document Deduplication

Upload multiple text or PDF documents

Automatically find near-duplicate documents with AI-powered similarity detection

Expand to view matching document content with similarity scores

Supports .txt and .pdf formats

📁 File Structure Dedupify/ │ ├── app.py # Main Streamlit app ├── requirements.txt # Required Python packages └── README.md # Project documentation 📦 Installation 

Clone the repository

git clone https://github.com/yourusername/dedupify.git cd dedupify 

Install dependencies

It’s recommended to use a virtual environment:

python -m venv venv source venv/bin/activate # or venv\Scripts\activate on Windows pip install -r requirements.txt 

Run the Streamlit app

streamlit run app.py 📊 Tech Stack 

Python 3.9+

Streamlit

scikit-learn (TF-IDF, clustering, cosine similarity)

PyPDF2 (PDF reading)

Pandas, NumPy, Matplotlib

✅ Example Inputs Text Clustering Example: Document about climate change and global warming. A brief on economic development in Southeast Asia. Machine learning techniques for clustering documents. An overview of renewable energy sources. Economic growth and trade policy impacts. Document Deduplication: 

Upload .txt or .pdf files such as:

report1.pdf, report2.pdf

article1.txt, article2.txt

🛡️ License 

This project is licensed under the MIT License.

🙌 Acknowledgments 

Developed with ❤️ by Prerna Gyanchandani
© 2025 All Rights Reserved.

