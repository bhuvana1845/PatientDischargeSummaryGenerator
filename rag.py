import os, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class MedicalRAG:
    def __init__(self, knowledge_folder="data/medical_knowledge", vectorstore_folder="vectorstore"):
        self.knowledge_folder = knowledge_folder
        self.vectorstore_folder = vectorstore_folder
        self.index_file = os.path.join(vectorstore_folder, "medical.index")
        self.documents_file = os.path.join(vectorstore_folder, "documents.pkl")
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []
        self.load_vector_store()

    def load_documents(self):
        documents = []
        if not os.path.exists(self.knowledge_folder):
            return documents
        for filename in os.listdir(self.knowledge_folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.knowledge_folder, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if text:
                        documents.append({"filename": filename, "text": text})
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        return documents

    def create_vector_store(self):
        self.documents = self.load_documents()
        if not self.documents:
            raise ValueError("No medical knowledge files found.")
        texts = [d["text"] for d in self.documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        os.makedirs(self.vectorstore_folder, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.documents_file, "wb") as f:
            pickle.dump(self.documents, f)

    def load_vector_store(self):
        if os.path.exists(self.index_file) and os.path.exists(self.documents_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.documents_file, "rb") as f:
                    self.documents = pickle.load(f)
            except Exception:
                self.index, self.documents = None, []

    def retrieve_documents(self, query, top_k=3):
        if self.index is None:
            raise ValueError("Vector database is not available. Run python ingest.py first.")
        if not query.strip():
            return []
        q = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(q, k)
        return [{"filename": self.documents[i]["filename"], "text": self.documents[i]["text"], "score": float(s)}
                for s, i in zip(scores[0], indices[0]) if i != -1]

def load_vector_store():
    return MedicalRAG()

def retrieve_documents(rag, query, top_k=3):
    return rag.retrieve_documents(query, top_k)
