# # Use an official Python runtime as a parent image
# FROM python:3.10-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements file first
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application's code
# COPY . .

# # Default command
# ENTRYPOINT ["python", "cli.py"]

# Use official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy required files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask application
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
