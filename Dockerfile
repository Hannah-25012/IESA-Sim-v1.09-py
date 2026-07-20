# Serves this project through the unified-project gateway (see app.py).
# Pinned to 3.9 (not the unified-project template's 3.12) to match the
# Python version this project's dependencies (see requirements.txt) have
# actually been run and verified against.
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
