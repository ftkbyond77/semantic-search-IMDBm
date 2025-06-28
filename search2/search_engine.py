import sqlite3
import numpy as np
from faker import Faker
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Faker for sample data
fake = Faker()

# Create and populate SQLite database
def create_database():
    db_path = '/app/data/documents.db'  # Persistent storage in Docker
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, title TEXT, content TEXT)''')
    
    # Generate 500 sample documents if table is empty
    c.execute("SELECT COUNT(*) FROM documents")
    if c.fetchone()[0] == 0:
        documents = []
        for i in range(500):
            title = fake.sentence(nb_words=6)
            content = fake.paragraph(nb_sentences=5)
            documents.append((title, content))
            c.execute("INSERT INTO documents (title, content) VALUES (?, ?)", (title, content))
    
        conn.commit()
    conn.close()
    return documents

# Preprocess text for BM25
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    return [token for token in tokens if token not in stop_words]

# Hybrid Search Engine class
class HybridSearchEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.document_ids = []
        self.bm25 = None
        self.faiss_index = None
        self.dimension = 384  # Dimension of MiniLM embeddings
        
    def load_data(self):
        conn = sqlite3.connect('/app/data/documents.db')
        c = conn.cursor()
        c.execute("SELECT id, title, content FROM documents")
        rows = c.fetchall()
        conn.close()
        
        self.documents = [f"{row[1]} {row[2]}" for row in rows]
        self.document_ids = [row[0] for row in rows]
        
        # Prepare BM25
        tokenized_corpus = [preprocess_text(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Prepare FAISS
        embeddings = self.model.encode(self.documents)
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index.add(np.array(embeddings).astype('float32'))
        
    def search(self, query, k=5, alpha=0.6):
        # BM25 scores
        tokenized_query = preprocess_text(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # FAISS search
        query_embedding = self.model.encode([query]).astype('float32')
        faiss_distances, faiss_indices = self.faiss_index.search(query_embedding, k)
        
        # Combine scores
        hybrid_scores = {}
        for idx, bm25_score in enumerate(bm25_scores):
            hybrid_scores[idx] = bm25_score * (1 - alpha)
        
        for i, idx in enumerate(faiss_indices[0]):
            if idx in hybrid_scores:
                hybrid_scores[idx] += (1.0 / (faiss_distances[0][i] + 1e-6)) * alpha
        
        # Sort results
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Retrieve results from database
        conn = sqlite3.connect('/app/data/documents.db')
        c = conn.cursor()
        results = []
        for idx, score in sorted_results:
            c.execute("SELECT id, title, content FROM documents WHERE id=?", (self.document_ids[idx],))
            doc = c.fetchone()
            results.append({
                'id': doc[0],
                'title': doc[1],
                'content': doc[2][:200] + "...",
                'score': score
            })
        conn.close()
        
        return results

def main():
    # Create and populate database
    create_database()
    
    # Initialize search engine
    search_engine = HybridSearchEngine()
    search_engine.load_data()
    
    # Example search
    query = "technology advancements in 2025"
    results = search_engine.search(query, k=5)
    
    print(f"Search results for: {query}")
    for result in results:
        print(f"\nID: {result['id']}")
        print(f"Title: {result['title']}")
        print(f"Content: {result['content']}")
        print(f"Score: {result['score']:.4f}")

if __name__ == "__main__":
    main()