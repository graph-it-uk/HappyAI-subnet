# Use the official Python 3.9 image as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Set the working directory in the container
WORKDIR /happy_ai/

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install dependencies
RUN pip install awscli

# Copy the rest of the application code
COPY . /happy_ai/

RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the FastAPI app will run on
EXPOSE 80

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
