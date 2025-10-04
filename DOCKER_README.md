# Docker Setup for Shorts AI Agent

This guide will help you run the Shorts AI Agent using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

## Quick Start

### 1. Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
# Copy the example and edit with your keys
cp .env.example .env
```

Edit `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
FAL_KEY=your_fal_api_key_here
FLASK_ENV=production
FLASK_DEBUG=0
```

### 2. Build and Run with Docker

#### Option A: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Run in background
docker-compose up -d --build
```

#### Option B: Using Docker directly

```bash
# Build the image
docker build -t shorts-ai-agent .

# Run the container
docker run -d \
  --name shorts-ai-agent \
  -p 5000:5000 \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/styles:/app/styles \
  shorts-ai-agent
```

### 3. Access the Application

Open your browser and go to: http://localhost:5000

## Docker Commands

### Build
```bash
docker build -t shorts-ai-agent .
```

### Run
```bash
docker run -d --name shorts-ai-agent -p 5000:5000 shorts-ai-agent
```

### Stop
```bash
docker stop shorts-ai-agent
```

### Remove
```bash
docker rm shorts-ai-agent
```

### View Logs
```bash
docker logs shorts-ai-agent
```

### Execute Commands in Container
```bash
docker exec -it shorts-ai-agent bash
```

## Docker Compose Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

### Rebuild and Start
```bash
docker-compose up --build -d
```

## Volume Mounts

The Docker setup includes the following volume mounts:

- `./videos:/app/videos` - Persists generated videos
- `./.env:/app/.env:ro` - Environment variables (read-only)
- `./styles:/app/styles` - Style images directory

## Health Check

The container includes a health check that verifies the application is running:
- Checks every 30 seconds
- Timeout: 10 seconds
- Retries: 3 times
- Start period: 40 seconds

## Resource Limits

Default resource limits (adjust in docker-compose.yml):
- Memory: 2GB limit, 512MB reservation
- CPU: 1.0 limit, 0.5 reservation

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs shorts-ai-agent

# Check if port is already in use
netstat -tulpn | grep :5000
```

### Permission Issues
```bash
# Fix permissions for mounted volumes
sudo chown -R $USER:$USER videos/ styles/
```

### API Key Issues
- Ensure `.env` file exists and contains valid API keys
- Check that the file is properly mounted in the container

### FFmpeg Issues
- FFmpeg is included in the Docker image
- If you encounter issues, check the container logs

## Production Deployment

For production deployment:

1. Use a reverse proxy (nginx) in front of the container
2. Set up SSL/TLS certificates
3. Use environment-specific `.env` files
4. Consider using Docker secrets for sensitive data
5. Set up monitoring and logging

## Example Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  shorts-ai-agent:
    build: .
    ports:
      - "127.0.0.1:5000:5000"  # Bind to localhost only
    volumes:
      - ./videos:/app/videos
      - ./.env.prod:/app/.env:ro
    environment:
      - FLASK_ENV=production
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```
