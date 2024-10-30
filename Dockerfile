# Start from a PyTorch image with the desired CUDA version if needed
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Set the working directory in the container
WORKDIR /app

# Copy requirements file to the container
COPY environment.yml /app/environment.yml

# Install dependencies via Conda
RUN conda env create -f /app/environment.yml

# Copy the project files to the container
COPY code/ .


# Expose a port if needed (e.g., for an API endpoint)
#EXPOSE 5000

# Run the project’s main script by default
CMD ["python", "laplace_transformation.py"]
