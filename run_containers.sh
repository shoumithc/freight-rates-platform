#!/bin/bash
# Build and run the FreightRates platform: one API + one UI on a shared Docker network
set -e

docker network create freight-net 2>/dev/null || true
# remove this platform's containers and those of the two earlier deployments
docker rm -f freight-api freight-ui freight-backend freight-frontend freight-rag freight-rag-ui 2>/dev/null || true

# Public address of the API for the Swagger link in the UI (Codespaces sets these variables)
API_PUBLIC_URL=""
if [ -n "$CODESPACE_NAME" ]; then
  API_PUBLIC_URL="https://${CODESPACE_NAME}-7860.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
fi

# API (Flask) - reachable from the UI as http://freight-api:7860
docker build -t freight-api ./api_files
docker run -d --name freight-api --network freight-net -p 7860:7860 --restart unless-stopped freight-api

# UI (Streamlit)
docker build -t freight-ui ./ui_files
docker run -d --name freight-ui --network freight-net -p 8501:8501 --restart unless-stopped \
  -e API_URL=http://freight-api:7860 -e API_PUBLIC_URL="$API_PUBLIC_URL" freight-ui

docker ps
echo ""
echo "API -> port 7860 (Swagger: add /docs) | App -> port 8501   (see the PORTS tab)"
echo "The forecast is ready at once; the assistant needs ~1 minute to load: curl localhost:7860/health"
