from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ragnexus.core.security import create_access_token, hash_password, verify_password
from ragnexus.infrastructure.database.models.user import User
from ragnexus.infrastructure.database.models.workspace import Workspace, WorkspaceMember
from ragnexus.config import Settings


class EmailAlreadyRegistered(Exception):
    pass


class InvalidCredentials(Exception):
    pass

class InactiveAccount(Exception):
    pass


def register_user(db: Session, *, email: str, password: str, settings: Settings) -> User:
    normalized_email = email.strip().lower()
    existing = db.scalar(select(User).where(User.email == normalized_email))
    if existing:
        raise EmailAlreadyRegistered

    user = User(email=normalized_email, password_hash=hash_password(password))
    try:
        db.add(user)
        db.flush()  # obtains user.id without committing the transaction
        workspace = Workspace(
            name="Personal Knowledge Base",
            workspace_type="personal",
            created_by=user.id,
        )
        db.add(workspace)
        db.flush()
        db.add(WorkspaceMember(
            workspace_id=workspace.id,
            user_id=user.id,
            role="owner",
        ))
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError as exc:
        db.rollback()
        # Covers concurrent registrations racing on the unique email index.
        raise EmailAlreadyRegistered from exc
    except Exception:
        db.rollback()
        raise


def authenticate_user(db: Session, *, email: str, password: str, settings: Settings) -> tuple[User, str]:
    normalized_email = email.strip().lower()
    user = db.scalar(select(User).where(User.email == normalized_email))
    # Use a dummy verification for unknown accounts to reduce timing differences.
    if user is None:
        verify_password(password, settings.DUMMY_PASSWORD_HASH)
        raise InvalidCredentials
    if not verify_password(password, user.password_hash):
        raise InvalidCredentials
    if not user.is_active:
        raise InactiveAccount
    token = create_access_token(str(user.id), settings)
    return user, token
