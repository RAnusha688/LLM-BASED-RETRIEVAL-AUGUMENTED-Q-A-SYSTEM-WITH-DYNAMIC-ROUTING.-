FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first
COPY requirements.txt .

# Install Python dependencies
# Install Python dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy all project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
