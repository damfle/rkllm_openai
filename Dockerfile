# Stage 1: Build stage
FROM python:3.13-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy source code
COPY . .

# Build the package
RUN pip wheel --no-deps --wheel-dir /build/wheels .

# Fetch RKLLM library from GitHub
RUN wget -O rknn-llm.zip https://github.com/airockchip/rknn-llm/archive/refs/tags/release-v1.2.2.zip && \
    unzip rknn-llm.zip && \
    mkdir -p /build/lib && \
    cp rknn-llm-release-v1.2.2/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so /build/lib/

# Stage 2: Runtime stage
FROM python:3.13-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 app

# Set working directory
WORKDIR /app

# Copy wheels and install
COPY --from=builder /build/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Copy RKLLM library
COPY --from=builder /build/lib/librkllmrt.so /usr/local/lib/
RUN ldconfig

# Change ownership to app user
RUN chown -R app:app /app

# Switch to app user
USER app

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Default command
CMD ["python", "-m", "rkllm_openai.server", "--host", "0.0.0.0", "--port", "8000"]
