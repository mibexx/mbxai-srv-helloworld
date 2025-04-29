import logging
import asyncio
import json
from typing import (
    Any,
    Optional,
    Callable,
    Awaitable,
    Type,
    Union,
    AsyncGenerator,
)

from openai import (
    AsyncOpenAI,
    OpenAIError,
)
from openai._types import NOT_GIVEN
from pydantic import BaseModel

from ..config import get_openrouter_api_config
from .models import OpenRouterModel


logger = logging.getLogger(__name__)


class OpenRouterApiClient:
    def __init__(
        self,
        model: OpenRouterModel = OpenRouterModel.GPT_41,
        max_retries: int = 3,
        retry_delay: int = 2,
    ) -> None:
        """
        Initializes the OpenRouterApiClient with the specified model.

        Args:
            model (OpenRouterModel): The model to use for API calls, defaults to GPT_41.
            max_retries (int): Maximum number of retry attempts for failed API calls, defaults to 3.
            retry_delay (int): Base delay between retries in seconds, defaults to 2.
        """
        openrouter_config = get_openrouter_api_config()

        logger.info(f"Initializing OpenRouter client with base URL: {openrouter_config.base_url}")
        self._client = AsyncOpenAI(
            base_url=openrouter_config.base_url,
            api_key=openrouter_config.api_key,
            timeout=90.0,
        )
        self.default_model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._registered_tools = {}
        self._tool_choice = "auto"
        self._tool_handlers = {}

    def register_tool(
        self, 
        tool_name: str, 
        tool_schema: dict[str, Any],
        handler: Optional[Callable[[dict[str, Any]], Awaitable[Any]]] = None
    ) -> None:
        """
        Register a tool with the OpenRouter client.
        
        Args:
            tool_name (str): The name of the tool to register.
            tool_schema (dict[str, Any]): The JSON schema for the tool.
            handler (Optional[Callable[[dict[str, Any]], Awaitable[Any]]]): Async function to handle tool calls.
        """
        self._registered_tools[tool_name] = tool_schema
        if handler:
            self._tool_handlers[tool_name] = handler
        logger.info(f"Registered tool: {tool_name}")

    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool from the OpenRouter client.
        
        Args:
            tool_name (str): The name of the tool to unregister.
        """
        if tool_name in self._registered_tools:
            del self._registered_tools[tool_name]
            if tool_name in self._tool_handlers:
                del self._tool_handlers[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
        else:
            logger.warning(f"Attempted to unregister non-existent tool: {tool_name}")

    def set_tool_choice(self, tool_choice: str) -> None:
        """
        Set the tool choice strategy for the client.
        
        Args:
            tool_choice (str): The tool choice strategy. Can be "auto", "none", or a specific tool name.
        """
        self._tool_choice = tool_choice
        logger.info(f"Set tool choice to: {tool_choice}")

    def get_registered_tools(self) -> dict[str, dict[str, Any]]:
        """
        Get all registered tools.
        
        Returns:
            dict[str, dict[str, Any]]: A dictionary of registered tools and their schemas.
        """
        return self._registered_tools.copy()

    async def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a registered tool with the given arguments.
        
        Args:
            tool_name (str): The name of the tool to execute.
            arguments (dict[str, Any]): The arguments to pass to the tool.
            
        Returns:
            Any: The result of the tool execution.
            
        Raises:
            ValueError: If the tool is not registered or has no handler.
        """
        if tool_name not in self._registered_tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
            
        if tool_name not in self._tool_handlers:
            raise ValueError(f"Tool '{tool_name}' has no handler registered")
            
        handler = self._tool_handlers[tool_name]
        return await handler(arguments)

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
        Process tool calls from a message and execute them.
        
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
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in message.tool_calls
            ]
            
            # Execute tools
            for tool_call in tool_calls:
                try:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    result = await self._execute_tool(tool_name, arguments)
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result)
                    })
                except Exception as e:
                    logger.error(f"Error executing tool {tool_call['function']['name']}: {e}")
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": tool_call["function"]["name"],
                        "content": f"Error: {str(e)}"
                    })
            
            # Add the assistant's message with tool calls
            updated_messages = messages.copy()
            updated_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            
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
        call_tools: bool = False,
        tool_choice: Optional[str] = None,
        max_iterations: int = 5,
    ) -> dict[str, Any]:
        """
        Unified method for interacting with the OpenRouter API.
        Automatically chooses between chat_completion and chat_parse based on whether structured_output is provided.
        
        Args:
            messages (list[dict[str, str]]): A list of messages to send to the chat.
            structured_output (Optional[Union[Type[BaseModel], dict[str, Any]]]): 
                - If a Pydantic model class is provided, it will be used for structured parsing.
                - If a dict is provided, it will be used as a response format for chat completion.
                - If None, a regular chat completion will be performed.
            model (Optional[OpenRouterModel]): The model to use for this request, defaults to the instance's default model.
            extra_headers (Optional[dict[str, str]]): Additional headers to include in the API request.
            call_tools (bool): Whether to include tool calls in the response, defaults to False.
            tool_choice (Optional[str]): Override the default tool choice strategy for this request.
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
        
        tools = None
        if call_tools and self._registered_tools:
            tools = [
                {"type": "function", "function": {"name": name, "parameters": schema}}
                for name, schema in self._registered_tools.items()
            ]
            
        tool_choice_param = tool_choice or self._tool_choice
        
        # Determine if we're using structured parsing or chat completion
        is_structured_parse = False
        response_format = None
        
        if structured_output is not None:
            if isinstance(structured_output, type) and issubclass(structured_output, BaseModel):
                # It's a Pydantic model, use it directly for structured parsing
                is_structured_parse = True
                structured_output_schema = structured_output
            elif isinstance(structured_output, dict):
                # It's a dict, use it as response format for chat completion
                response_format = structured_output
            else:
                raise ValueError("structured_output must be a Pydantic model class or a dict")
        
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
                structured_output=structured_output_schema if is_structured_parse else None,
                tools=tools,
                tool_choice=tool_choice_param,
            )
            
            # Extract the message from the response
            message = response.choices[0].message
            
            # Process tool calls if present
            tool_calls, tool_results, updated_messages = await self._process_tool_calls(message, current_messages)
            
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
                "tool_results": all_tool_results
            }
        else:
            return {
                "content": message.content or "",
                "tool_calls": all_tool_calls,
                "tool_results": all_tool_results
            }

    async def agent_stream(
        self,
        messages: list[dict[str, str]],
        structured_output: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        model: Optional[OpenRouterModel] = None,
        extra_headers: Optional[dict[str, str]] = None,
        call_tools: bool = False,
        tool_choice: Optional[str] = None,
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
            call_tools (bool): Whether to include tool calls in the response, defaults to False.
            tool_choice (Optional[str]): Override the default tool choice strategy for this request.
            max_iterations (int): Maximum number of iterations for the agent loop, defaults to 5.
            
        Yields:
            dict[str, Any]: A dictionary containing the current state of the agent.
        """
        model = model or self.default_model
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            **(extra_headers or {}),
        }
        
        tools = None
        if call_tools and self._registered_tools:
            tools = [
                {"type": "function", "function": {"name": name, "parameters": schema}}
                for name, schema in self._registered_tools.items()
            ]
            
        tool_choice_param = tool_choice or self._tool_choice
        
        # Determine if we're using structured parsing or chat completion
        is_structured_parse = False
        response_format = None
        
        if structured_output is not None:
            if isinstance(structured_output, type) and issubclass(structured_output, BaseModel):
                # It's a Pydantic model, use it directly for structured parsing
                is_structured_parse = True
                structured_output_schema = structured_output
            elif isinstance(structured_output, dict):
                # It's a dict, use it as response format for chat completion
                response_format = structured_output
            else:
                raise ValueError("structured_output must be a Pydantic model class or a dict")
        
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
                structured_output=structured_output_schema if is_structured_parse else None,
                tools=tools,
                tool_choice=tool_choice_param,
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
                    "is_final": False
                }
            else:
                yield {
                    "content": message.content or "",
                    "tool_calls": [],
                    "tool_results": [],
                    "iteration": iteration,
                    "is_final": False
                }
            
            # Process tool calls if present
            tool_calls, tool_results, updated_messages = await self._process_tool_calls(message, current_messages)
            
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
                        "is_final": False
                    }
                else:
                    yield {
                        "content": message.content or "",
                        "tool_calls": tool_calls,
                        "tool_results": tool_results,
                        "iteration": iteration,
                        "is_final": False
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
                "is_final": True
            }
        else:
            yield {
                "content": message.content or "",
                "tool_calls": all_tool_calls,
                "tool_results": all_tool_results,
                "iteration": iteration,
                "is_final": True
            }

    async def chat_parse(
        self,
        messages: list[dict[str, str]],
        structured_output: Union[Type[BaseModel], dict[str, Any]],
        model: Optional[OpenRouterModel] = None,
        extra_headers: Optional[dict[str, str]] = None,
        call_tools: bool = False,
        tool_choice: Optional[str] = None,
        use_agent: bool = False,
    ) -> dict[str, Any]:
        """
        Sends a chat completion request to the API and returns the structured response content.

        Args:
            messages (list[dict[str, str]]): A list of messages to send to the chat.
            structured_output (Union[Type[BaseModel], dict[str, Any]]): The schema defining the structure of the response.
            model (Optional[OpenRouterModel]): The model to use for this request, defaults to the instance's default model.
            extra_headers (Optional[dict[str, str]]): Additional headers to include in the API request.
            call_tools (bool): Whether to include tool calls in the response, defaults to False.
            tool_choice (Optional[str]): Override the default tool choice strategy for this request.
            use_agent (bool): Whether to execute tools when they are called, defaults to False.

        Returns:
            dict[str, Any]: A dictionary containing the parsed response, tool calls, and tool results if any.
        """
        # Pass the structured_output directly without conversion
        return await self.agent(
            messages=messages,
            structured_output=structured_output,
            model=model,
            extra_headers=extra_headers,
            call_tools=call_tools,
            tool_choice=tool_choice,
            max_iterations=1 if not use_agent else 5,
        )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[OpenRouterModel] = None,
        extra_headers: Optional[dict[str, str]] = None,
        structured_output: Optional[dict[str, Any]] = None,
        call_tools: bool = False,
        tool_choice: Optional[str] = None,
        use_agent: bool = False,
    ) -> dict[str, Any]:
        """
        Sends a chat completion request to the API and returns the response content.

        Args:
            messages (list[dict[str, str]]): A list of messages to send to the chat.
            model (Optional[OpenRouterModel]): The model to use for this request, defaults to the instance's default
            model.
            extra_headers (Optional[dict[str, str]]): Additional headers to include in the API request, defaults to
            an empty dict.
            structured_output (Optional[dict[str, Any]]): Specifies the desired structure of the response,
            defaults to NOT_GIVEN.
            call_tools (bool): Whether to include tool calls in the response, defaults to False.
            tool_choice (Optional[str]): Override the default tool choice strategy for this request.
            use_agent (bool): Whether to execute tools when they are called, defaults to False.

        Returns:
            dict[str, Any]: A dictionary containing the content, tool calls, and tool results if any.
        """
        return await self.agent(
            messages=messages,
            structured_output=structured_output,
            model=model,
            extra_headers=extra_headers,
            call_tools=call_tools,
            tool_choice=tool_choice,
            max_iterations=1 if not use_agent else 5,
        ) 