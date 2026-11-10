from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ragnexus.application.auth.service import (
    EmailAlreadyRegistered, InactiveAccount, InvalidCredentials,
    authenticate_user, register_user,
)
from ragnexus.config import Settings
from ragnexus.dependencies import get_app_settings, get_current_user, get_db
from ragnexus.infrastructure.database.models.user import User
from ragnexus.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserRead

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db), settings: Settings = Depends(get_app_settings)):
    try:
        return register_user(db, email=str(payload.email), password=payload.password, settings=settings)
    except EmailAlreadyRegistered:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db), settings: Settings = Depends(get_app_settings)):
    try:
        user, token = authenticate_user(db, email=str(payload.email), password=payload.password, settings=settings)
    except InvalidCredentials:
        raise HTTPException(status_code=401, detail="Incorrect email or password.", headers={"WWW-Authenticate": "Bearer"})
    except InactiveAccount:
        raise HTTPException(status_code=403, detail="This account is inactive.")
    return TokenResponse(
        access_token=token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user,
    )


@router.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)):
    return current_user
