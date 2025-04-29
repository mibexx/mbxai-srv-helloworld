FROM mibexx/mbxai-service:latest

# Copy application code
COPY . .

# Install dependencies
RUN uv sync --no-dev