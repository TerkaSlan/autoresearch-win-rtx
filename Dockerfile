FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Use bash as default shell instead of dash
RUN echo "dash dash/sh boolean false" | debconf-set-selections && \
    dpkg-reconfigure dash

# Symlink /bin/sh to /bin/bash
RUN rm -f /bin/sh && ln -s /bin/bash /bin/sh

# Install Python 3.10+, build tools, and other packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    git \
    tmux \
    curl \
    vim \
    wget \
    build-essential \
    ninja-build \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set CUDA arch flags for Blackwell GPU (sm_100)
ENV TORCH_CUDA_ARCH_LIST="100"
ENV CUDA_NVCC_FLAGS="-gencode=arch=compute_100,code=sm_100"
ENV FORCE_CUDNN_VERSION_MAJOR=9

# Create non-root user
RUN useradd -m -u 1000 autoresearch

# Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    cp /root/.local/bin/uvx /usr/local/bin/uvx && \
    chmod +x /usr/local/bin/uv /usr/local/bin/uvx

# Add uv to global PATH
ENV PATH="/usr/local/bin:${PATH}"

# Create working directory
RUN mkdir -p /workspace

# Set working directory
WORKDIR /workspace/autoresearch-sdpa

# Copy project files (instead of cloning from GitHub)
COPY --chown=autoresearch:autoresearch train.py .
COPY --chown=autoresearch:autoresearch prepare.py .
COPY --chown=autoresearch:autoresearch pyproject.toml .
COPY --chown=autoresearch:autoresearch program.md .

# Install dependencies using uv (as root)
RUN uv pip install --system torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install --system -r <(uv pip compile pyproject.toml)

# Create checkpoints directory
RUN mkdir -p checkpoints

# Chown everything to autoresearch user
RUN chown -R autoresearch:autoresearch /workspace

# Switch to autoresearch user
USER autoresearch

# Default command - can be overridden
CMD ["/bin/bash", "-c", "sleep infinity"]