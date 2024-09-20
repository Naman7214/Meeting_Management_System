from flask import Flask, render_template, request, redirect, url_for, flash
import os
from config import GEMINI_API_KEY
from extensions import db

app = Flask(__name__)
app.config.from_pyfile('config.py')
db.init_app(app)

from models.models import Meeting, Participant, DiscussionPoint, Document

# Import utility functions
from utils.rag_utils import match_discussion_points
from utils.llm_utils import (
    generate_agenda,
    generate_summary_with_rag,
    collection,  # Chroma collection
    embedding_model  # SentenceTransformer model
)

from utils.transcript_utils import transcribe_media_with_gemini

@app.route('/')
def index():
    meetings = Meeting.query.all()
    return render_template('index.html', meetings=meetings)

@app.route('/create_meeting', methods=['GET', 'POST'])
def create_meeting():
    if request.method == 'POST':
        title = request.form['title']
        if Meeting.query.filter_by(title=title).first():
            flash('Meeting with this title already exists. Please choose a different title.')
            return redirect(url_for('create_meeting'))
        meeting = Meeting(title=title)
        db.session.add(meeting)
        db.session.commit()
        return redirect(url_for('meeting', meeting_id=meeting.id))
    return render_template('create_meeting.html')

@app.route('/meeting/<int:meeting_id>')
def meeting(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    return render_template('meeting.html', meeting=meeting)

@app.route('/meeting/<int:meeting_id>/upload_document', methods=['POST'])
def upload_document(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    file = request.files['document']
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], meeting.title, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        doc = Document(filename=filename, filepath=filepath, meeting_id=meeting.id)
        db.session.add(doc)
        db.session.commit()

        # Read document content
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading document {filename}: {e}")
            content = ""

        # Generate embedding
        embedding = embedding_model.encode(content).tolist()

        # Add to Chroma
        collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[{
                "meeting_id": meeting.id,
                "type": "document",
                "filename": filename
            }],
            ids=[f"doc_{doc.id}"]
        )

    return redirect(url_for('meeting', meeting_id=meeting.id))

@app.route('/meeting/<int:meeting_id>/add_point', methods=['POST'])
def add_point(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    participant_name = request.form['name']
    content = request.form['point']
    participant = Participant.query.filter_by(name=participant_name).first()
    if not participant:
        participant = Participant(name=participant_name)
        db.session.add(participant)
        db.session.commit()
    point = DiscussionPoint(content=content, participant_id=participant.id, meeting_id=meeting.id)
    db.session.add(point)
    db.session.commit()

    # Generate embedding
    embedding = embedding_model.encode(content).tolist()

    # Add to Chroma
    collection.add(
        documents=[content],
        embeddings=[embedding],
        metadatas=[{
            "meeting_id": meeting.id,
            "type": "discussion_point",
            "participant": participant_name
        }],
        ids=[f"dp_{point.id}"]
    )

    return redirect(url_for('meeting', meeting_id=meeting.id))

@app.route('/meeting/<int:meeting_id>/generate_agenda')
def generate_agenda_route(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    points = [dp.content for dp in meeting.discussion_points]

    # Concatenate discussion points to form a query
    query_text = ' '.join(points)

    # Query ChromaDB for relevant document content
    results = collection.query(
        query_texts=[query_text],
        n_results=5,  # Adjust as needed
        where={
            "$and": [
                {"meeting_id": meeting.id},
                {"type": "document"}
            ]
        },
        include=['documents']
    )

    # Extract relevant document content
    if results['documents']:
        relevant_docs = results['documents'][0]
    else:
        relevant_docs = []

    # Generate the agenda with discussion points and relevant documents
    agenda = generate_agenda(points, relevant_docs)

    return render_template('agenda.html', agenda=agenda, meeting=meeting)

@app.route('/meeting/<int:meeting_id>/upload_recording', methods=['POST'])
def upload_recording(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    file = request.files['recording']
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], meeting.title, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)

        # Transcribe media (audio/video) using Gemini
        transcript = transcribe_media_with_gemini(filepath)

        # Match discussion points
        points = [dp.content for dp in meeting.discussion_points]
        addressed, unaddressed = match_discussion_points(transcript, points)

        # Update database
        for dp in meeting.discussion_points:
            if dp.content in addressed:
                dp.addressed = True
        db.session.commit()

        # Save transcript
        transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], meeting.title, 'transcript.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)

        # Split transcript into sentences
        transcript_sentences = transcript.split('.')

        # Generate embeddings for each sentence
        embeddings = embedding_model.encode(transcript_sentences).tolist()

        # Add to Chroma
        collection.add(
            documents=transcript_sentences,
            embeddings=embeddings,
            metadatas=[{
                "meeting_id": meeting.id,
                "type": "transcript",
                "sentence_index": idx
            } for idx in range(len(transcript_sentences))],
            ids=[f"transcript_{meeting.id}_{idx}" for idx in range(len(transcript_sentences))]
        )

        return redirect(url_for('generate_summary', meeting_id=meeting.id))
    else:
        return "No file uploaded", 400

@app.route('/meeting/<int:meeting_id>/generate_summary')
def generate_summary(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    # Load transcript
    transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], meeting.title, 'transcript.txt')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read()

    # Get addressed and unaddressed points
    addressed_points = [dp.content for dp in meeting.discussion_points if dp.addressed]
    unaddressed_points = [dp.content for dp in meeting.discussion_points if not dp.addressed]

    # Generate summary with RAG
    summary = generate_summary_with_rag(transcript, addressed_points, unaddressed_points, meeting)

    return render_template('summary.html', summary=summary, meeting=meeting)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)