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
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class VectorStore:
    def __init__(self):
        print("Initializing vector store...")
        self.index_name = os.getenv('PINECONE_INDEX_NAME')
        pc = Pinecone(api_key=os.getenv("PINECONCE_API_KEY"))
        self.index = pc.Index(self.index_name)

        
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 
        print("Vector store initialized")

    def embed(self, text: str) -> list[float]:
            return self.model(tf.constant([text])).numpy().tolist()[0]

    def add(self, data: str, namespace: str, metadata: dict) -> bool:
        val = self.embed(data)
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
        values = self.embed(query)

        response = self.index.query(
            top_k=3,
            vector=[values],
            namespace=namespace,
            include_metadata=True,
            include_values=False
        )['matches']

        texts = [r['metadata']['text'] for r in response]
        return "\n\n".join(texts)


if __name__ == "__main__":
    vector_store = VectorStore()
    