FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY squid/ /app/squid/

CMD ["python", "-m", "squid.agents.tier1"]
