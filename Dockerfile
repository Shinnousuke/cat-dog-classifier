# Use an official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy all files from current directory to /app in container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir streamlit tensorflow pillow numpy

# Expose Streamlit’s default port
EXPOSE 8501

# Set environment variables to avoid Streamlit prompts
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# Command to run the Streamlit app
CMD ["streamlit", "run", "application.py"]
