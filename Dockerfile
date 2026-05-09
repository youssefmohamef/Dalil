# 1. Use a lightweight Python base image to reduce image size
FROM python:3.9-slim

# 2. Install system dependencies required for FAISS and Torch
RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /code

# 4. Copy requirements first to leverage Docker cache for faster builds
COPY ./requirements.txt /code/requirements.txt

# 5. Install Python dependencies (Torch CPU-only to save space)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy the rest of the project files to the container
COPY . .

# 7. Grant full read/write permissions for vector_db_index storage
RUN chmod -R 777 /code

# 8. Run the FastAPI server on port 7860 (required by Hugging Face Spaces)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]