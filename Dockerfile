FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install build dependencies (gcc needed for some python packages)
RUN apt-get update && apt-get install -y build-essential gcc

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader punkt stopwords

# Copy the rest of the application
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Command to run both (using a script or supervisor would be better, but keeping it simple)
# For now, let's just default to running the API. The user can override CMD.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
