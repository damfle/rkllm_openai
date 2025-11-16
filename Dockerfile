# Stage 1: Build stage
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy source code
COPY . .

# Build the package
RUN pip wheel --no-deps --wheel-dir /build/wheels .

# Fetch and build RKNN toolkit from official repository
RUN wget -O rknn-toolkit2.zip https://github.com/rockchip-linux/rknn-toolkit2/archive/refs/heads/master.zip && \
    unzip rknn-toolkit2.zip && \
    cd rknn-toolkit2-master && \
    mkdir -p /build/rknn && \
    cp -r rknpu2/runtime/Linux/librknn_api/aarch64/* /build/rknn/ && \
    cp -r rknpu2/runtime/Linux/librknn_api/include/* /build/rknn/ && \
    cd .. && \
    rm -rf rknn-toolkit2.zip rknn-toolkit2-master

# Fetch RKLLM library from airockchip repo
RUN wget -O rknn-llm.zip https://github.com/airockchip/rknn-llm/archive/refs/tags/release-v1.2.2.zip && \
    unzip rknn-llm.zip && \
    mkdir -p /build/rkllm && \
    cp rknn-llm-release-v1.2.2/rkllm-runtime/Linux/librkllm_api/aarch64/* /build/rkllm/ && \
    rm -rf rknn-llm.zip rknn-llm-release-v1.2.2


# Stage 2: Runtime stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    nfs-common \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 app

# Set working directory
WORKDIR /app

# Copy wheels and install
COPY --from=builder /build/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Copy RKNN libraries from build stage
COPY --from=builder /build/rknn/*.so /usr/local/lib/
COPY --from=builder /build/rknn/*.h /usr/local/include/
# Copy RKLLM libraries from build stage
COPY --from=builder /build/rkllm/* /usr/lib/
RUN ldconfig

# Set file limits like install.sh does
RUN echo "* soft nofile 16384" >> /etc/security/limits.conf && \
    echo "* hard nofile 1048576" >> /etc/security/limits.conf && \
    echo "root soft nofile 16384" >> /etc/security/limits.conf && \
    echo "root hard nofile 1048576" >> /etc/security/limits.conf

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
