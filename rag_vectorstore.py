
import uuid
import chromadb
from sentence_transformers import SentenceTransformer


def build_vector_store(chunks):
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.Client()

    collection = client.create_collection(
        name=f"pdf_{uuid.uuid4().hex[:12]}"
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    ).tolist()

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"page": int(chunk["page"])}
            for chunk in chunks
        ]
    )

    return embedding_model, client, collection
