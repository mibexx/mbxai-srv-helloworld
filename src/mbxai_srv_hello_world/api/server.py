from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Any

from ..config import get_config
from .definition import MbxDefinitionBuilder
from ..project.api import router as project_router

# Initialize the FastAPI app
app = FastAPI(
    title="MBXAI SRV Hello World",
    description="An AI Service that can be launched in the Mibexx AI Orchestrator to expose ai features",
)

# Include project routers
app.include_router(project_router)


class IdentResponse(BaseModel):
    """Identity response model."""

    name: str
    version: str


@app.get("/ident", response_model=IdentResponse)
async def ident() -> IdentResponse:
    """Return basic service identity information."""
    config = get_config()
    return IdentResponse(
        name=config.name,
        version=config.version,
    )


@app.get("/mbxai-definition")
async def mbxai_definition() -> List[Dict[str, Any]]:
    """Return the API definition for all endpoints.

    This endpoint generates documentation about all API endpoints
    including their paths, methods, request schemas, and response schemas.
    Endpoints '/ident' and '/mbxai-definition' are excluded.
    """
    excluded_paths = {"/ident", "/mbxai-definition"}
    builder = MbxDefinitionBuilder(app=app, excluded_paths=excluded_paths)
    return builder.build_definitions()
