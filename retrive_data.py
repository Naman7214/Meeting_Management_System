# retrieve_data.py

import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(
    path="chroma_db",  # Replace "test" with your actual path if different
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Get the collection
collection = chroma_client.get_or_create_collection(name="meeting_embeddings")

# Retrieve all items in the collection
results = collection.get(include=['documents', 'metadatas', 'embeddings'])

print(f"Number of items in collection: {len(results['ids'])}\n")

for doc_id, doc, metadata, embedding in zip(
    results['ids'], results['documents'], results['metadatas'], results['embeddings']
):
    print(f"ID: {doc_id}")
    print(f"Document: {doc}")
    print(f"Metadata: {metadata}")
    print(f"Embedding Length: {len(embedding)}")
    print("-----")

# # Optionally, perform a sample query
# # Initialize embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Define embedding function
# def embedding_function(texts):
#     return embedding_model.encode(texts).tolist()

# # Sample query text
# query_text = "Discuss project timelines and deadlines."

# # Perform the query
# results = collection.query(
#     query_texts=[query_text],
#     n_results=5,
#     include=['documents', 'metadatas', 'distances'],
#     embedding_function=embedding_function  # Provide the embedding function
# )

# print("\nQuery Results:")
# for doc_id, doc, metadata, distance in zip(
#     results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0]
# ):
#     print(f"ID: {doc_id}")
#     print(f"Document: {doc}")
#     print(f"Metadata: {metadata}")
#     print(f"Distance: {distance}")
#     print("-----")
