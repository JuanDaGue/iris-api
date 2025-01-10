# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variable for model path
ENV MODEL_PATH=models/data/processed/model.pkl

# Expose the FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
