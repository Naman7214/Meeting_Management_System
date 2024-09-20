import os
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# Initialize Chroma client with persistent storage
chroma_client = chromadb.PersistentClient(
    path="chroma_db",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)
# Create or get a collection
collection = chroma_client.get_or_create_collection(name="meeting_embeddings")

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_agenda(points, relevant_docs):
    # Prepare discussion points text
    points_text = ''
    for idx, point in enumerate(points, 1):
        points_text += f"{idx}. {point}\n"

    # Prepare relevant documents text
    docs_text = ''
    if relevant_docs:
        docs_text += 'Relevant Document Content:\n'
        for doc in relevant_docs:
            # Optionally limit the length of the document content
            excerpt = doc[:500]  # Limit to first 500 characters
            docs_text += f"{excerpt}\n\n"

    prompt = f"""
Using the following discussion points and relevant document content, generate a detailed meeting agenda with clear headings and subheadings.

Discussion Points:
{points_text}

{docs_text}

Provide the agenda in a structured format suitable for a professional meeting.
"""

    # Choose the Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Generate agenda using Gemini
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=800,
            temperature=0.5,
        ),
    )

    agenda = response.text.strip()
    return agenda


def generate_summary_with_rag(transcript, addressed_points, unaddressed_points, meeting):
    # Generate embedding for the transcript
    query_embedding = embedding_model.encode(transcript).tolist()

    # Query Chroma for relevant documents (without meeting_id filter)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=25  # Fetch more results for broader context
    )
    print(results)
    # Extract relevant documents and metadatas
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    # Build context
    context = ''
    for doc, metadata in zip(documents, metadatas):
        meeting_indicator = "Current Meeting" if metadata.get('meeting_id') == meeting.id else "Other Meeting"
        context += f"{metadata['type'].capitalize()} ({meeting_indicator}): {doc}\n"

    prompt = f"""
Using the following context, meeting transcript, and reference documents, generate a detailed summary including topics discussed, key decisions made, assigned action items with responsible participants, and highlight any unresolved issues.

Context:
{context}

Transcript:
{transcript}

Addressed Discussion Points:
{addressed_points}

Unaddressed Discussion Points:
{unaddressed_points}

Provide the summary in a clear and structured format.
"""

    # Generate summary using Gemini
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(
        [prompt],
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.5,
            max_output_tokens=1500
        ),
        request_options={"timeout": 600},
    )

    summary = response.text.strip()
    return summary

