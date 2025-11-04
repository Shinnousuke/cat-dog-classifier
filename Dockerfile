# Use an official TensorFlow image (includes Python, TensorFlow, and dependencies)
FROM tensorflow/tensorflow:2.13.0

# Set working directory inside the container
WORKDIR /app

# Copy all files from your project into the container
COPY . /app

# Install any extra Python libraries if needed
# (you can skip this if everything is already included in TensorFlow)
RUN pip install --no-cache-dir numpy pandas matplotlib

# Create a directory inside the container for saving models
RUN mkdir -p /app/saved_models

# Set environment variable to avoid interactive prompts
ENV TF_CPP_MIN_LOG_LEVEL=2

# Command to run your Python script
CMD ["python", "train_model.py"]
