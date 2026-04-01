#!/bin/bash
# HiveResearch — one-command local dev setup
set -e

echo "=== HiveResearch Dev Setup ==="

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Python 3.11+ required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js 18+ required (for MCP client testing)"; }

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install
source .venv/bin/activate
pip install -e ".[gpu,bio]" 2>/dev/null || pip install -e .

# Copy .env if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env — fill in your API keys"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Start services:"
echo "  docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d"
echo ""
echo "Or run coordinator locally:"
echo "  source .venv/bin/activate"
echo "  uvicorn hive.coordinator.app:app --reload --port 8000"
echo ""
echo "Health check:"
echo "  curl http://localhost:8000/health"
