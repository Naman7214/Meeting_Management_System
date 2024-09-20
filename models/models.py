from extensions import db

class Meeting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), unique=True, nullable=False)
    documents = db.relationship('Document', backref='meeting', lazy=True)
    discussion_points = db.relationship('DiscussionPoint', backref='meeting', lazy=True)

class Participant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    discussion_points = db.relationship('DiscussionPoint', backref='participant', lazy=True)

class DiscussionPoint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text)
    participant_id = db.Column(db.Integer, db.ForeignKey('participant.id'), nullable=False)
    meeting_id = db.Column(db.Integer, db.ForeignKey('meeting.id'), nullable=False)
    addressed = db.Column(db.Boolean, default=False)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100))
    filepath = db.Column(db.String(200))
    meeting_id = db.Column(db.Integer, db.ForeignKey('meeting.id'), nullable=False)
