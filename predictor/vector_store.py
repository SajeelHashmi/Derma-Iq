"""
This file will serve to create vector store with pine cone.
"""
import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
import tensorflow_hub as hub
import tensorflow as tf 
import logging
import fitz  # PyMuPDF
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class VectorStore:

    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Skip initialization if already initialized
        print
        if VectorStore._initialized:
            print("VectorStore already initialized.")
            return

        print("Initializing vector store...")
        self.index_name = os.getenv('PINECONE_INDEX_NAME')
        print(f"Index name: {self.index_name}")
        pc = Pinecone(api_key=os.getenv("PINECONCE_API_KEY"))
        self.index = pc.Index(self.index_name)

        
        self.model = hub.load(r"H:\tf_sentence_encoder\universal-sentence-encoder-tensorflow2-universal-sentence-encoder-v2") 
        print("Vector store initialized")
        VectorStore._initialized = True

    def embed(self, text: str)->list[float] :
            embed =self.model([text])
            return embed[0].numpy().tolist()  
    
    def chunk_text(self, text: str, chunk_size=500, overlap=100) -> list[str]:
        sentences = sent_tokenize(text)
        chunks = []
        chunk = ""

        for sentence in sentences:
            if len(chunk) + len(sentence) <= chunk_size:
                chunk += " " + sentence
            else:
                chunks.append(chunk.strip())
                # Start new chunk with some overlap
                chunk = " ".join(chunk.strip().split()[-overlap:]) + " " + sentence

        if chunk:
            chunks.append(chunk.strip())
        return chunks

    def add_pdf(self, pdf_path: str, namespace: str =None, meta_base: dict = None):
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        chunks = self.chunk_text(full_text)
        logging.info(f"PDF has been split into {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            metadata = meta_base.copy() if meta_base else {}
            metadata["chunk_index"] = i
            self.add(chunk, namespace=namespace, metadata=metadata)

    def test_search_query(self, query: str, namespace: str):
        print(f"Searching for: {query}")
        result = self.search(query, namespace)
        print("Retrieved text:")
        print(result)

    def add(self, data: str, namespace: str, metadata: dict) -> bool:
        val = self.embed(data)
        print(f"Embedding vector: {type(val)} ,{type(val[0])}")
        metadata["text"] = data

        vectors = [{
            "id": str(uuid.uuid4()),
            "values": val,
            "metadata": metadata
        }]

        self.index.upsert(vectors=vectors, namespace=namespace)
        print("Data added to vector store")
        return True

    def search(self, query: str, namespace: str) -> str:
        try:
            values = self.embed(query)
            print(f"Query vector: {values}")
            response = self.index.query(
                top_k=5,
                vector=values,
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )['matches']

            texts = [r['metadata']['text'] for r in response]
            return "\n\n".join(texts)
        except Exception as e:
            logging.error(f"Error in search: {e}")
            return ""

# def test():
#     # this should not reinitialize the vector store
#     vector_store = VectorStore()
#     test_data = "This is a test data"
#     embed = vector_store.embed(test_data)
#     print(f"Embedding for '{test_data}': {embed}")

if __name__ == "__main__":
    vector_store = VectorStore()
    test_data = "This is a test data"
    embed = vector_store.embed(test_data)
    print(f"Embedding for '{test_data}': {len(embed)}")

    vector_store.add_pdf("H:\FYP WEB APP\docs\SKIN_Diseases_PDF.pdf","general", {"source": "SKIN_Diseases_PDF.pdf"})
    vector_store.test_search_query("What is the treatment for acne?", "general")
