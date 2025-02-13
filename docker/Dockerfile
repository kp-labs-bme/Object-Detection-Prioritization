# Base Image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Metadata
WORKDIR /workspace
LABEL maintainer="your.email@example.com"
LABEL docker_image_name="ML development"
LABEL description="This container is created for ML development"

# System settings
RUN echo "fs.inotify.max_user_watches = 524288" >> /etc/sysctl.conf
RUN sysctl -p --system

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Budapest

# Install necessary packages with retries
RUN apt-get update &&  \
    apt-get install -y --no-install-recommends \
    tzdata \
    openssh-server \
    libgl1 \
    libpq-dev \
    cmake \
    vim \
    sudo \
    git \
    wget \
    curl \
    ssh \
    htop \
    tmux \
    rsync \
    x11-apps \
    ffmpeg \
    graphviz \
    python3-tk \
    nano \
    unzip \
    bzip2 \
    build-essential \
    autoconf \
    automake \
    libjpeg-dev \
    zlib1g-dev

# Clean up
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Set environment variables for NVIDIA compatibility
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN echo "PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" >> /etc/environment
RUN echo "PYTHONPATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" >> /etc/environment

# Copy requirements.txt
COPY ./requirements.txt /requirements.txt

# Install dependencies excluding torch and torchvision
RUN pip install -r /requirements.txt

# Clear any existing torch installations
RUN pip uninstall -y torch torchvision

RUN pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
RUN pip install torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html

ENTRYPOINT ["/entry.sh"]