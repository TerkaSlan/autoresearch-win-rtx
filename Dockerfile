FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Use bash as default shell instead of dash
RUN echo "dash dash/sh boolean false" | debconf-set-selections && \
    dpkg-reconfigure dash

# Symlink /bin/sh to /bin/bash
RUN rm -f /bin/sh && ln -s /bin/bash /bin/sh

# Install Python 3.10+ and build tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    curl \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create non-root user first
RUN useradd -m -u 1000 inference

# Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    cp /root/.local/bin/uvx /usr/local/bin/uvx && \
    chmod +x /usr/local/bin/uv /usr/local/bin/uvx

# Add uv to global PATH
ENV PATH="/usr/local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=inference:inference model.py .
COPY --chown=inference:inference inference.py .
COPY --chown=inference:inference api_server.py .
COPY --chown=inference:inference pyproject.toml .

# Install dependencies using uv
RUN uv pip install --system torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install --system -r <(uv pip compile pyproject.toml --no-dev)

# Create checkpoints directory
RUN mkdir -p /app/checkpoints && chown inference:inference /app/checkpoints

# Switch to non-root user
USER inference

# Expose API port
EXPOSE 8000

# Default command - run the FastAPI server
CMD ["uv", "run", "python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]