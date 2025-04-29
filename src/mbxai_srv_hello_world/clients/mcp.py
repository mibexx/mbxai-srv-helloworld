import json
import logging
import asyncio
import httpx
from typing import (
    Any,
    Optional,
    Callable,
    Awaitable,
    Type,
    Union,
    AsyncGenerator,
)

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..config import get_openrouter_api_config
from .models import OpenRouterModel


logger = logging.getLogger(__name__)


class McpClient:
    """
    A client for the Model Context Protocol (MCP) that can connect to MCP servers,
    discover tools, and use them with an LLM.
    """

    def __init__(
        self,
        model: OpenRouterModel = OpenRouterModel.GPT_41,
        max_retries: int = 3,
        retry_delay: int = 2,
    ) -> None:
        """
        Initializes the McpClient with the specified model.

        Args:
            model (OpenRouterModel): The model to use for API calls, defaults to GPT_41.
            max_retries (int): Maximum number of retry attempts for failed API calls, defaults to 3.
            retry_delay (int): Base delay between retries in seconds, defaults to 2.
        """
        openrouter_config = get_openrouter_api_config()

        logger.info(
            f"Initializing OpenRouter client with base URL: {openrouter_config.base_url}"
        )
        self._client = AsyncOpenAI(
            base_url=openrouter_config.base_url,
            api_key=openrouter_config.api_key,
            timeout=90.0,
        )
        self.default_model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # MCP-specific attributes
        self._mcp_clients: dict[str, ClientSession] = (
            {}
        )  # Map of tool name to MCP client
        self._available_tools: list[dict[str, Any]] = []  # List of available tools
        self._messages: list[dict[str, Any]] = []  # Conversation history
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def add_mcp_server(
        self, server_params: Union[StdioServerParameters, str]
    ) -> None:
        """
        Connect to an MCP server and discover its tools.

        Args:
            server_params (Union[StdioServerParameters, str]):
                - If StdioServerParameters: Parameters for connecting to a local MCP server via stdio
                - If str: URL for connecting to a remote MCP server via HTTP
        """
        if isinstance(server_params, str):
            await self.add_http_mcp_server(server_params)
        else:
            await self.add_stdio_mcp_server(server_params)

    async def add_stdio_mcp_server(self, server_params: StdioServerParameters) -> None:
        """
        Connect to a local MCP server via stdio and discover its tools.

        Args:
            server_params (StdioServerParameters): Parameters for connecting to the MCP server.
        """
        try:
            # Connect to MCP server
            async with stdio_client(server_params) as (read_stream, write_stream):
                mcp = ClientSession(name="mbxai-mcp-client", version="1.0.0")
                await mcp.connect(read_stream, write_stream)

                # Get tools from server
                tools_result = await mcp.list_tools()

                logger.info(
                    f"Connected to MCP server with tools: {[tool.name for tool in tools_result.tools]}"
                )

                # Add tools to available tools
                for tool in tools_result.tools:
                    self._mcp_clients[tool.name] = mcp
                    self._available_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            },
                        }
                    )
        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}")
            raise

    async def add_http_mcp_server(self, server_url: str) -> None:
        """
        Connect to a remote MCP server via HTTP and discover its tools.

        Args:
            server_url (str): URL of the MCP server.
        """
        try:
            # Fetch tools from the server
            async with self._http_client as client:
                response = await client.get(f"{server_url.rstrip('/')}/api/tools")
                response.raise_for_status()
                tools_data = response.json()

                logger.info(
                    f"Connected to MCP server with tools: {[tool['name'] for tool in tools_data['tools']]}"
                )

                # Add tools to available tools
                for tool in tools_data["tools"]:
                    self._available_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["inputSchema"],
                            },
                        }
                    )
                    # Store the internal URL for direct invocation
                    self._mcp_clients[tool["name"]] = {
                        "internal_url": tool["internal_url"]
                    }
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            raise

    async def invoke_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Directly invoke a tool by name with the given parameters.

        Args:
            tool_name (str): Name of the tool to invoke
            **kwargs: Parameters to pass to the tool

        Returns:
            Any: The tool's response

        Raises:
            ValueError: If the tool is not found
        """
        if tool_name not in self._mcp_clients:
            raise ValueError(f"Tool {tool_name} not found")

        tool_info = self._mcp_clients[tool_name]
        if isinstance(tool_info, dict) and "internal_url" in tool_info:
            # External MCP server tool
            async with self._http_client as client:
                response = await client.post(tool_info["internal_url"], json=kwargs)
                response.raise_for_status()
                return response.json()
        else:
            # Local MCP server tool
            return await self._execute_with_retry(
                "invoke_tool", tool_info.invoke_tool, tool_name, kwargs
            )

    def get_available_tools(self) -> list[dict[str, Any]]:
        """
        Get all available tools from connected MCP servers.

        Returns:
            list[dict[str, Any]]: A list of available tools.
        """
        return self._available_tools.copy()

    async def _execute_with_retry(
        self, operation_name: str, operation_func, *args, **kwargs
    ):
        """
        Execute an operation with retry logic.

        Args:
            operation_name: Name of the operation for logging
            operation_func: Async function to execute
            *args, **kwargs: Arguments to pass to the function

        Returns:
            The result of the operation or None if all retries failed
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Executing {operation_name} (attempt {attempt}/{self.max_retries})"
                )

                # Use asyncio.wait_for to implement timeout
                result = await asyncio.wait_for(
                    operation_func(*args, **kwargs), timeout=90.0
                )

                return result

            except asyncio.TimeoutError:
                logger.warning(
                    f"{operation_name} timed out after 30 seconds (attempt {attempt}/{self.max_retries})"
                )
                if attempt == self.max_retries:
                    logger.error(
                        f"All {self.max_retries} attempts for {operation_name} timed out"
                    )
                    return None

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parsing error in {operation_name}: {e} (attempt {attempt}/{self.max_retries})"
                )
                if attempt == self.max_retries:
                    logger.error(
                        f"All {self.max_retries} attempts for {operation_name} failed with JSON parsing errors"
                    )
                    return None

            except OpenAIError as e:
                logger.warning(
                    f"OpenAI API Error in {operation_name}: {e} (attempt {attempt}/{self.max_retries})"
                )
                if attempt == self.max_retries:
                    logger.error(
                        f"All {self.max_retries} attempts for {operation_name} failed with OpenAI API errors"
                    )
                    return None

            except Exception as e:
                logger.warning(
                    f"Unexpected error in {operation_name}: {e} (attempt {attempt}/{self.max_retries})"
                )
                if attempt == self.max_retries:
                    logger.error(
                        f"All {self.max_retries} attempts for {operation_name} failed with unexpected errors"
                    )
                    return None

            # Wait before retrying
            await asyncio.sleep(self.retry_delay * attempt)  # Exponential backoff
            logger.info(f"Retrying {operation_name}...")

    async def _process_tool_calls(self, message, messages):
        """
        Process tool calls from a message and execute them using MCP clients.

        Args:
            message: The message containing tool calls.
            messages: The current conversation messages.

        Returns:
            tuple: (tool_calls, tool_results, updated_messages)
        """
        tool_calls = []
        tool_results = []

        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]

            # Execute tools using MCP clients
            for tool_call in tool_calls:
                try:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])

                    # Get the appropriate MCP client for this tool
                    mcp_client = self._mcp_clients.get(tool_name)
                    if mcp_client:
                        result = await mcp_client.call_tool(
                            {"name": tool_name, "arguments": tool_args}
                        )
                        tool_results.append(
                            {
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": tool_name,
                                "content": result.content[0].text,
                            }
                        )
                    else:
                        logger.error(f"No MCP client found for tool: {tool_name}")
                        tool_results.append(
                            {
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": tool_name,
                                "content": f"Error: No MCP client found for tool: {tool_name}",
                            }
                        )
                except Exception as e:
                    logger.error(
                        f"Error executing tool {tool_call['function']['name']}: {e}"
                    )
                    tool_results.append(
                        {
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": tool_call["function"]["name"],
                            "content": f"Error: {str(e)}",
                        }
                    )

            # Add the assistant's message with tool calls
            updated_messages = messages.copy()
            updated_messages.append(
                {"role": "assistant", "content": None, "tool_calls": tool_calls}
            )

            # Add tool results to messages
            for result in tool_results:
                updated_messages.append(result)

            return tool_calls, tool_results, updated_messages

        return tool_calls, tool_results, messages

    async def _call_llm(
        self,
        model_value: str,
        messages: list[dict[str, Any]],
        extra_headers: dict[str, str],
        response_format=None,
        structured_output=None,
        tools=None,
        tool_choice=None,
    ):
        """
        Call the LLM with the given parameters.

        Args:
            model_value: The model to use.
            messages: The messages to send.
            extra_headers: Additional headers.
            response_format: The response format for chat completion.
            structured_output: The structured output for chat parse.
            tools: The tools to use.
            tool_choice: The tool choice strategy.

        Returns:
            The response from the LLM.
        """
        if structured_output is not None:
            # Use chat parse
            params = {
                "model": model_value,
                "messages": messages,
                "response_format": structured_output,
                "extra_headers": extra_headers,
            }

            if tools:
                params["tools"] = tools
                if tool_choice:
                    params["tool_choice"] = tool_choice

            return await self._client.beta.chat.completions.parse(**params)
        else:
            # Use chat completion
            params = {
                "model": model_value,
                "messages": messages,
                "extra_headers": extra_headers,
            }

            if response_format:
                params["response_format"] = response_format

            if tools:
                params["tools"] = tools
                if tool_choice:
                    params["tool_choice"] = tool_choice

            return await self._client.chat.completions.create(**params)

    async def agent(
        self,
        messages: list[dict[str, str]],
        structured_output: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        model: Optional[OpenRouterModel] = None,
        extra_headers: Optional[dict[str, str]] = None,
        max_iterations: int = 5,
    ) -> dict[str, Any]:
        """
        Run an agent that can use tools from connected MCP servers.

        Args:
            messages (list[dict[str, str]]): A list of messages to send to the chat.
            structured_output (Optional[Union[Type[BaseModel], dict[str, Any]]]):
                - If a Pydantic model class is provided, it will be used for structured parsing.
                - If a dict is provided, it will be used as a response format for chat completion.
                - If None, a regular chat completion will be performed.
            model (Optional[OpenRouterModel]): The model to use for this request, defaults to the instance's default model.
            extra_headers (Optional[dict[str, str]]): Additional headers to include in the API request.
            max_iterations (int): Maximum number of iterations for the agent loop, defaults to 5.

        Returns:
            dict[str, Any]: A dictionary containing the response content, parsed response (if structured_output is a Pydantic model),
                          tool calls, and tool results if any.
        """
        model = model or self.default_model
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            **(extra_headers or {}),
        }

        # Determine if we're using structured parsing or chat completion
        is_structured_parse = False
        response_format = None

        if structured_output is not None:
            if isinstance(structured_output, type) and issubclass(
                structured_output, BaseModel
            ):
                # It's a Pydantic model, use it directly for structured parsing
                is_structured_parse = True
                structured_output_schema = structured_output
            elif isinstance(structured_output, dict):
                # It's a dict, use it as response format for chat completion
                response_format = structured_output
            else:
                raise ValueError(
                    "structured_output must be a Pydantic model class or a dict"
                )

        # Initialize variables for the agent loop
        current_messages = messages.copy()
        all_tool_calls = []
        all_tool_results = []
        iteration = 0

        # MCP-style agent loop
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{max_iterations}")

            # Call the LLM
            response = await self._call_llm(
                model.value,
                current_messages,
                headers,
                response_format=response_format,
                structured_output=(
                    structured_output_schema if is_structured_parse else None
                ),
                tools=self._available_tools,
                tool_choice="auto",
            )

            # Extract the message from the response
            message = response.choices[0].message

            # Process tool calls if present
            tool_calls, tool_results, updated_messages = await self._process_tool_calls(
                message, current_messages
            )

            # Update the current messages
            current_messages = updated_messages

            # Add tool calls and results to the accumulated lists
            all_tool_calls.extend(tool_calls)
            all_tool_results.extend(tool_results)

            # If there are no tool calls, we're done
            if not tool_calls:
                break

        # Prepare the final result
        if is_structured_parse:
            return {
                "parsed": message.parsed,
                "tool_calls": all_tool_calls,
                "tool_results": all_tool_results,
            }
        else:
            return {
                "content": message.content or "",
                "tool_calls": all_tool_calls,
                "tool_results": all_tool_results,
            }

    async def agent_stream(
        self,
        messages: list[dict[str, str]],
        structured_output: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        model: Optional[OpenRouterModel] = None,
        extra_headers: Optional[dict[str, str]] = None,
        max_iterations: int = 5,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream the agent's responses, yielding each step of the process.

        Args:
            messages (list[dict[str, str]]): A list of messages to send to the chat.
            structured_output (Optional[Union[Type[BaseModel], dict[str, Any]]]):
                - If a Pydantic model class is provided, it will be used for structured parsing.
                - If a dict is provided, it will be used as a response format for chat completion.
                - If None, a regular chat completion will be performed.
            model (Optional[OpenRouterModel]): The model to use for this request, defaults to the instance's default model.
            extra_headers (Optional[dict[str, str]]): Additional headers to include in the API request.
            max_iterations (int): Maximum number of iterations for the agent loop, defaults to 5.

        Yields:
            dict[str, Any]: A dictionary containing the current state of the agent.
        """
        model = model or self.default_model
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            **(extra_headers or {}),
        }

        # Determine if we're using structured parsing or chat completion
        is_structured_parse = False
        response_format = None

        if structured_output is not None:
            if isinstance(structured_output, type) and issubclass(
                structured_output, BaseModel
            ):
                # It's a Pydantic model, use it directly for structured parsing
                is_structured_parse = True
                structured_output_schema = structured_output
            elif isinstance(structured_output, dict):
                # It's a dict, use it as response format for chat completion
                response_format = structured_output
            else:
                raise ValueError(
                    "structured_output must be a Pydantic model class or a dict"
                )

        # Initialize variables for the agent loop
        current_messages = messages.copy()
        all_tool_calls = []
        all_tool_results = []
        iteration = 0

        # MCP-style agent loop with streaming
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{max_iterations}")

            # Call the LLM
            response = await self._call_llm(
                model.value,
                current_messages,
                headers,
                response_format=response_format,
                structured_output=(
                    structured_output_schema if is_structured_parse else None
                ),
                tools=self._available_tools,
                tool_choice="auto",
            )

            # Extract the message from the response
            message = response.choices[0].message

            # Yield the current state
            if is_structured_parse:
                yield {
                    "parsed": message.parsed,
                    "tool_calls": [],
                    "tool_results": [],
                    "iteration": iteration,
                    "is_final": False,
                }
            else:
                yield {
                    "content": message.content or "",
                    "tool_calls": [],
                    "tool_results": [],
                    "iteration": iteration,
                    "is_final": False,
                }

            # Process tool calls if present
            tool_calls, tool_results, updated_messages = await self._process_tool_calls(
                message, current_messages
            )

            # Update the current messages
            current_messages = updated_messages

            # Add tool calls and results to the accumulated lists
            all_tool_calls.extend(tool_calls)
            all_tool_results.extend(tool_results)

            # Yield the tool calls and results
            if tool_calls:
                if is_structured_parse:
                    yield {
                        "parsed": message.parsed,
                        "tool_calls": tool_calls,
                        "tool_results": tool_results,
                        "iteration": iteration,
                        "is_final": False,
                    }
                else:
                    yield {
                        "content": message.content or "",
                        "tool_calls": tool_calls,
                        "tool_results": tool_results,
                        "iteration": iteration,
                        "is_final": False,
                    }

            # If there are no tool calls, we're done
            if not tool_calls:
                break

        # Yield the final result
        if is_structured_parse:
            yield {
                "parsed": message.parsed,
                "tool_calls": all_tool_calls,
                "tool_results": all_tool_results,
                "iteration": iteration,
                "is_final": True,
            }
        else:
            yield {
                "content": message.content or "",
                "tool_calls": all_tool_calls,
                "tool_results": all_tool_results,
                "iteration": iteration,
                "is_final": True,
            }
