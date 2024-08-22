# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including libsndfile and ffmpeg
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY backend/ .  

# Copy the outputs directory into the container
COPY outputs /app/outputs  

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Use the entrypoint to bind the correct port
ENTRYPOINT ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} app:app"]