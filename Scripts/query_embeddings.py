import os
import openai
from pinecone import Pinecone

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("urata-soft")

def get_similar_chunks(question, namespace="nextgen-specs"):
    embedding = openai.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True,
        namespace=namespace
    )

    return [match["metadata"]["text"] for match in results["matches"]]
