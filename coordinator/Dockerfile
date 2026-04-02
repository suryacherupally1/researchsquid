FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY squid/ /app/squid/

EXPOSE 8000

CMD ["uvicorn", "squid.coordinator.app:app", "--host", "0.0.0.0", "--port", "8000"]
