from fastapi import APIRouter

from ragnexus.api.v1.health import router as health_router
from ragnexus.api.v1.endpoints.auth import router as auth_router
from ragnexus.api.v1.endpoints.workspaces import router as workspaces_router
from ragnexus.api.v1.endpoints.ingestion import router as ingestion_router
from ragnexus.api.v1.endpoints.agent_approvals import router as agent_approvals_router
from ragnexus.api.v1.endpoints.chat_router_factory import create_chat_router
from ragnexus.dependencies import get_current_user
from ragnexus.rag.orchestration.bootstrap import build_orchestration_service
from ragnexus.api.v1.orchestration_proxy import LazyOrchestrationService

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(workspaces_router)
api_router.include_router(ingestion_router)
api_router.include_router(agent_approvals_router)
api_router.include_router(
    create_chat_router(
        get_current_user=get_current_user,
        orchestration=LazyOrchestrationService(build_orchestration_service),
    )
)
