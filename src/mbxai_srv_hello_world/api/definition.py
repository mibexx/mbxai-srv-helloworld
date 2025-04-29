"""API definition builder module."""

import inspect
from typing import Any, Dict, List, Optional, Set, Type

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from pydantic import BaseModel, create_model


class MbxDefinitionBuilder:
    """Builder for API definition information.
    
    This class extracts information about API endpoints including:
    - Path
    - HTTP Method
    - Request schema
    - Response schema
    
    It excludes specified endpoints from the definition.
    """
    
    def __init__(self, app: FastAPI, excluded_paths: Optional[Set[str]] = None):
        """Initialize the definition builder.
        
        Args:
            app: The FastAPI application
            excluded_paths: Optional set of paths to exclude from the definition
        """
        self.app = app
        self.excluded_paths = excluded_paths or set()
        # Always exclude the definition endpoint itself
        self.excluded_paths.add("/mbxai-definition")
        
    def _extract_model_schema(self, model_type: Optional[Type]) -> Optional[Dict[str, Any]]:
        """Extract JSON schema from a Pydantic model.
        
        Args:
            model_type: The model type to extract schema from
            
        Returns:
            A JSON schema dictionary or None if model_type is not a Pydantic model
        """
        if model_type is None:
            return None
            
        if hasattr(model_type, "model_json_schema"):
            # Pydantic v2+
            return model_type.model_json_schema()
        elif hasattr(model_type, "schema"):
            # Pydantic v1 fallback
            return model_type.schema()
            
        # Not a Pydantic model
        return None

    def _get_request_model(self, route: APIRoute) -> Optional[Type[BaseModel]]:
        """Extract request model from a route.
        
        Args:
            route: The API route
            
        Returns:
            The request model type or None if not found
        """
        if not route.body_field:
            return None
            
        # Get the model from the body field
        if hasattr(route.body_field, "type_"):
            return route.body_field.type_
            
        return None
        
    def _get_response_model(self, route: APIRoute) -> Optional[Type[BaseModel]]:
        """Extract response model from a route.
        
        Args:
            route: The API route
            
        Returns:
            The response model type or None if not found
        """
        if route.response_model:
            return route.response_model
            
        return None
    
    def build_definitions(self) -> List[Dict[str, Any]]:
        """Build definitions for all endpoints except excluded ones.
        
        Returns:
            A list of endpoint definitions
        """
        definitions = []
        
        # Process all routes
        for route in self.app.routes:
            if not isinstance(route, APIRoute):
                continue
                
            # Skip excluded paths
            if route.path in self.excluded_paths:
                continue
                
            # Extract request and response models
            request_model = self._get_request_model(route)
            response_model = self._get_response_model(route)
            
            # Create definition
            definition = {
                "endpoint": route.path,
                "method": route.methods.copy().pop() if route.methods else "GET",
                "request_schema": self._extract_model_schema(request_model),
                "response_schema": self._extract_model_schema(response_model),
            }
            
            definitions.append(definition)
            
        return definitions