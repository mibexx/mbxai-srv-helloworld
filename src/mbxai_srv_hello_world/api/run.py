import uvicorn
import logging
import argparse
from pathlib import Path

from ..config import get_config

config = get_config()

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description=f"Run the {config.name} API server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload on code changes"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )

    args = parser.parse_args()

    logger.info(f"Starting {config.name} v{config.version} on {args.host}:{args.port}")

    uvicorn.run(
        "src.mbxai_srv_hello_world.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main() 