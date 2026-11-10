from typing import List

from sqlalchemy.orm import Session

from database.connection import SessionLocal
from database.models import Conversation


class LongTermMemory:
    def __init__(self):
        pass

    def save_message(self,session_id: str,user_id: str,role: str,message: str,) -> None:

        db: Session = SessionLocal()

        try:
            conversation = Conversation(
                session_id=session_id,
                user_id=user_id,
                role=role,
                message=message,
                message_type="chat",
                message_metadata=None
            )

            db.add(conversation)
            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def get_session_history(self,session_id: str,) -> List[Conversation]:


        db: Session = SessionLocal()

        try:
            conversations = (
                db.query(Conversation)
                .filter(
                    Conversation.session_id == session_id
                )
                .order_by(
                    Conversation.created_at.asc()
                )
                .all()
            )

            return[{"role": conv.role, "content": conv.message} for conv in conversations]

        finally:
            db.close()

    def clear_session(self,session_id: str,) -> None:
        
        db: Session = SessionLocal()

        try:
            (
                db.query(Conversation)
                .filter(
                    Conversation.session_id == session_id
                )
                .delete()
            )

            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def get_recent_messages(self,session_id: str,limit: int = 10,) -> List[Conversation]:

        db: Session = SessionLocal()

        try:
            conversations = (
                db.query(Conversation)
                .filter(
                    Conversation.session_id == session_id
                )
                .order_by(
                    Conversation.created_at.desc()
                )
                .limit(limit)
                .all()
            )

            return [{"role": conv.role,"content": conv.message} for conv in reversed(conversations)]

        finally:
            db.close()

"""
Handles persistent conversation storage in PostgreSQL.
"""