"""
feedback.py
------------
Feedback handling logic for search queries and results.
Builds on top of the UserFeedback ORM model.
"""

from datetime import datetime
from sqlalchemy.orm import Session
from .models import UserFeedback


def record_feedback(db: Session,
                    query_id: int,
                    user_id: str,
                    feedback_type: str,
                    feedback_value: str = None,
                    rating: int = None,
                    comment: str = None) -> UserFeedback:
    """
    Record a new piece of feedback for a given query.

    Args:
        db: SQLAlchemy session
        query_id: ID of the query the feedback relates to
        user_id: User providing feedback
        feedback_type: Type of feedback (e.g. "thumbs_up", "thumbs_down", "rating")
        feedback_value: Optional descriptive string
        rating: Optional numeric rating
        comment: Optional free-text comment

    Returns:
        The created UserFeedback ORM object
    """
    fb = UserFeedback(
        query_id=query_id,
        user_id=user_id,
        feedback_type=feedback_type,
        feedback_value=feedback_value,
        rating=rating,
        created_at=datetime.utcnow(),
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb


def get_feedback_for_query(db: Session, query_id: int):
    """
    Retrieve all feedback entries for a given query.
    """
    return db.query(UserFeedback).filter_by(query_id=query_id).all()


def average_rating_for_query(db: Session, query_id: int) -> float:
    """
    Compute the average numeric rating for a query.
    """
    ratings = [f.rating for f in get_feedback_for_query(db, query_id) if f.rating is not None]
    if not ratings:
        return 0.0
    return sum(ratings) / len(ratings)
