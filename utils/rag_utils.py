from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def match_discussion_points(transcript, discussion_points):
    point_embeddings = model.encode(discussion_points)
    transcript_sentences = transcript.split('.')
    transcript_embeddings = model.encode(transcript_sentences)

    addressed_points = []
    unaddressed_points = discussion_points.copy()

    for idx, point_embedding in enumerate(point_embeddings):
        similarities = cosine_similarity([point_embedding], transcript_embeddings)[0]
        max_similarity = np.max(similarities)
        if max_similarity >= 0.5:  # Adjust threshold as needed
            addressed_points.append(discussion_points[idx])
            unaddressed_points.remove(discussion_points[idx])

    return addressed_points, unaddressed_points
