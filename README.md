# MIBEXX AI Service Template

A template for creating AI services with FastAPI, Pydantic, SQLAlchemy, and Docker support.

## Features

- FastAPI for high-performance API development
- Pydantic for data validation and settings management
- SQLAlchemy for database operations
- Docker support for containerization
- OpenRouter API client for AI model interactions
- Model Context Protocol (MCP) client for tool discovery and execution
- Comprehensive testing setup with pytest
- Code quality tools: mypy, ruff
- GitHub Actions for CI/CD

## Project Structure

The template creates a well-organized project structure:

```
mbxai_srv_hello_world/
├── src/
│   └── mbxai_srv_hello_world/
│       ├── api/                  # Core API functionality
│       │   ├── definition.py     # API definition generation
│       │   ├── run.py            # Server startup
│       │   └── server.py         # FastAPI server configuration
│       ├── clients/              # Client libraries
│       │   ├── mcp.py            # Model Context Protocol client
│       │   ├── models.py         # Model definitions
│       │   └── openrouter.py     # OpenRouter API client
│       ├── project/              # Project-specific code
│       │   └── api.py            # Project API endpoints
│       ├── config.py             # Configuration management
│       └── __init__.py           # Package initialization
├── tests/                        # Test suite
├── data/                         # Data storage
├── logs/                         # Log files
├── kubernetes/                   # Kubernetes deployment files
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── pyproject.toml                # Project metadata and dependencies
└── README.md                     # This file
```

## Usage

### Creating a New Project

```bash
cookiecutter gh:mibexx/mbxai-srv-template
```

Follow the prompts to configure your project.

### Running the Service

```bash
# Install dependencies
pip install -e .

# Run the service
python -m src.mbxai_srv_hello_world.api.run
```

Or with command-line arguments:

```bash
python -m src.mbxai_srv_hello_world.api.run --host 127.0.0.1 --port 5000 --reload
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t mbxai_srv_hello_world .

# Run the container
docker run -p 5000:5000 mbxai_srv_hello_world
```

Or with docker-compose:

```bash
docker-compose up
```

### API Endpoints

The template includes several API endpoints:

- `GET /ident`: Returns basic service identity information
- `GET /mbxai-definition`: Returns the definition of all API endpoints
- Project-specific endpoints in the `/api` path

### AI Clients

The template includes two AI clients for different use cases:

1. **OpenRouter API Client** (`openrouter.py`): A direct client for the OpenRouter API that supports chat completions, structured output parsing, and tool execution.

2. **Model Context Protocol Client** (`mcp.py`): A client that implements the Model Context Protocol for tool discovery and execution, providing a more standardized approach to tool handling.

Both clients can be used independently or together, depending on your needs.

### OpenRouter API Client

The OpenRouter API client supports:

- Chat completions
- Structured output parsing
- Tool registration and execution
- Agent mode with multiple rounds of tool calls
- Streaming agent responses

#### Basic Usage

```python
from mbxai.clients.openrouter import OpenRouterApiClient, OpenRouterModel

# Initialize the client
client = OpenRouterApiClient()

# Send a message
response = await client.chat_completion(
    messages=[{"role": "user", "content": "Hello, world!"}]
)
print(response["content"])
```

#### Structured Output

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Parse structured output
response = await client.chat_parse(
    messages=[{"role": "user", "content": "My name is John and I am 30 years old."}],
    structured_output=UserInfo
)
print(response["parsed"])  # UserInfo(name="John", age=30)
```

#### Tool Registration

```python
async def search_database(query: str) -> str:
    # Implement database search
    return f"Results for: {query}"

# Register a tool
client.register_tool(
    name="search_database",
    description="Search the database for information",
    function=search_database,
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
)
```

#### Agent Mode

```python
# Run an agent that can use tools
response = await client.agent(
    messages=[{"role": "user", "content": "Find information about John"}],
    max_iterations=5
)
print(response["content"])
print(response["tool_calls"])
print(response["tool_results"])
```

#### Streaming Agent Responses

```python
# Stream agent responses
async for step in client.agent_stream(
    messages=[{"role": "user", "content": "Find information about John"}]
):
    if step["is_final"]:
        print("Final response:", step["content"])
    else:
        print(f"Iteration {step['iteration']}:", step["content"])
        if step["tool_calls"]:
            print("Tool calls:", step["tool_calls"])
            print("Tool results:", step["tool_results"])
```

### Model Context Protocol (MCP) Client

The MCP client supports:

- Connecting to MCP servers
- Discovering tools from servers
- Executing tools through the MCP protocol
- Agent mode with MCP tools
- Streaming agent responses

#### Basic Usage

```python
from your_package.clients.mcp import McpClient
from mcp import StdioServerParameters

# Initialize the client
client = McpClient()

# Connect to a local MCP server
server_params = StdioServerParameters(
    command=["python", "path/to/your/mcp_server.py"]
)
await client.add_mcp_server(server_params)

# Get available tools
tools = client.get_available_tools()
print("Available tools:", [tool["function"]["name"] for tool in tools])
```

#### Agent Mode with MCP Tools

```python
# Run an agent that can use MCP tools
response = await client.agent(
    messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
    max_iterations=5
)
print(response["content"])
print(response["tool_calls"])
print(response["tool_results"])
```

#### Streaming Agent Responses

```python
# Stream agent responses
async for step in client.agent_stream(
    messages=[{"role": "user", "content": "What's the weather like in Paris?"}]
):
    if step["is_final"]:
        print("Final response:", step["content"])
    else:
        print(f"Iteration {step['iteration']}:", step["content"])
        if step["tool_calls"]:
            print("Tool calls:", step["tool_calls"])
            print("Tool results:", step["tool_results"])
```

### Creating a Simple Tool

Here's an example of how to create a simple tool using the MCP approach:

#### 1. Create an MCP Server

```python
from mcp import Server, StdioServerTransport, Tool

# Create a server
server = Server("my-server")

# Define a tool
@server.tool()
async def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."

# Run the server
async def main():
    transport = StdioServerTransport()
    await server.serve(transport)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. Use the Tool with the MCP Client

```python
from mcp import StdioServerParameters

# Create server parameters
server_params = StdioServerParameters(
    command=["python", "path/to/your/mcp_server.py"]
)

# Run an agent that uses the weather tool
response = await client.agent(
    messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
    max_iterations=3
)
print(response["content"])
```

#### 3. Create a Tool with the OpenRouter Client

```python
from mbxai.clients.openrouter import OpenRouterApiClient

# Initialize the client
client = OpenRouterApiClient()

# Define the tool handler
async def get_weather(query: str) -> str:
    location = query
    # In a real implementation, you would call a weather API
    return json.dumps({
        "location": location,
        "temperature": 22,
        "condition": "Sunny",
        "humidity": 65
    })

# Register the tool
client.register_tool(
    name="get_weather",
    description="Get the current weather for a location",
    function=get_weather,
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city and country, e.g. 'London, UK'"}
        },
        "required": ["location"]
    }
)

# Run an agent that uses the weather tool
response = await client.agent(
    messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
    max_iterations=3
)
print(response["content"])
```

## Configuration

The service can be configured using environment variables with the prefix `MBXAI_SRV_HELLO_WORLD_`:

```
MBXAI_SRV_HELLO_WORLD_NAME=Custom Service Name
MBXAI_SRV_HELLO_WORLD_LOG_LEVEL=10  # DEBUG
MBXAI_SRV_HELLO_WORLD_OPENROUTER_API_KEY=your_api_key_here
```

## Development

### Testing

```bash
pytest
```

### Linting

```bash
ruff check .
```

### Type Checking

```bash
mypy src
```

## Requirements

- Python 3.12+
- Cookiecutter 2.5.0+

## License

This project is licensed under the MIT License - see the LICENSE file for details.
