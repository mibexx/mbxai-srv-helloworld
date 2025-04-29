"""Project-level API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from ..clients.mcp import McpClient
from ..config import get_mcp_config

# Create a router for project-level endpoints
router = APIRouter(prefix="/api", tags=["api"])


class HelloRequest(BaseModel):
    """Hello world request model."""
    name: str = Field("World", description="Name to greet")


class HelloResponse(BaseModel):
    """Hello world response model."""
    message: str
    name: str


class WeatherRequest(BaseModel):
    """Weather request model."""
    location: str = Field(..., description="Location to get weather for")


class WeatherResponse(BaseModel):
    """Weather response model."""
    weather_info: str


@router.post("/hello", response_model=HelloResponse)
async def hello_world(request: HelloRequest) -> HelloResponse:
    """A simple hello world endpoint.
    
    This is an example of a project-level endpoint that is automatically
    included in the main API.
    """
    return HelloResponse(
        message=f"Hello, {request.name}!",
        name=request.name,
    )


@router.post("/weather", response_model=WeatherResponse)
async def get_weather(request: WeatherRequest) -> WeatherResponse:
    """Get weather information for a location using OpenAI with MCP tools.
    
    This endpoint uses the MCP client to connect to an MCP server and
    uses OpenAI to process the weather request using available tools.
    """
    try:
        mcp_config = get_mcp_config()
        if not mcp_config.server_url:
            raise HTTPException(
                status_code=500,
                detail="MCP server URL not configured. Set MCP_SERVER_URL environment variable."
            )

        # Initialize MCP client and connect to server
        client = McpClient()
        await client.add_http_mcp_server(mcp_config.server_url)

        # Create the prompt for OpenAI
        messages = [
            {
                "role": "user",
                "content": f"Give me the weather for {request.location}"
            }
        ]

        # Use the agent to get weather information
        response = await client.agent(
            messages=messages,
            max_iterations=3  # Limit iterations since we just need one tool call
        )

        return WeatherResponse(weather_info=response["content"])

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get weather: {str(e)}") 