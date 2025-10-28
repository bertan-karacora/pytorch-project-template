ARG VERSION_UBUNTU_CONTAINER=24.04 \
    URL_KEYRING_NVIDIA_CONTAINER=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    VERSION_CUDA_CONTAINER=12-8 \
    VERSION_CUDNN_CONTAINER=9-cuda-12 \
    VERSION_PYTHON_CONTAINER=3.12 \
    VERSION_PIP_CONTAINER=25.2 \
    VERSION_SETUPTOOLS_CONTAINER=80.9.0 \
    VERSION_WHEEL_CONTAINER=0.45.1



# ========================================
FROM ubuntu:"$VERSION_UBUNTU_CONTAINER" AS base



# ========================================
FROM base AS system

# Set shell
SHELL ["/bin/bash", "-c"]

# Configure shell
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS="yes"

# Install essentials
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    apt-utils \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    nano \
    software-properties-common \
    tmux \
    unzip \
    wget \
    x11-apps \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Enable GPU
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Install CUDA Toolkit
ARG URL_KEYRING_NVIDIA_CONTAINER=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    VERSION_CUDA_CONTAINER=12-8
RUN wget "$URL_KEYRING_NVIDIA_CONTAINER" --quiet --directory-prefix /tmp/ && \
    dpkg --install /tmp/cuda-keyring_1.1-1_all.deb && \
    rm /tmp/cuda*.deb
RUN apt-get update --quiet && \
    apt-get install --quiet --assume-yes --no-install-recommends \
    "cuda-toolkit-$VERSION_CUDA_CONTAINER" \
    && \
    apt-get autoremove --quiet --assume-yes && \
    apt-get clean --quiet && \
    rm /var/lib/apt/lists/* --recursive --force
ENV PATH="/usr/local/cuda/bin:$PATH"
RUN ldconfig

# Install cuDNN
ARG VERSION_CUDNN_CONTAINER=9-cuda-12
RUN apt-get update --quiet && \
    apt-get install --quiet --assume-yes --no-install-recommends \
    "cudnn$VERSION_CUDNN_CONTAINER" \
    && \
    apt-get autoremove --quiet --assume-yes && \
    apt-get clean --quiet && \
    rm /var/lib/apt/lists/* --recursive --force

# Add group and user
RUN deluser --remove-home ubuntu || true && \
    delgroup ubuntu || true
ARG USER=user \
    UID=1001
RUN addgroup --gid "$UID" "$USER" && \
    adduser --uid "$UID" --ingroup users --disabled-password --gecos "" "$USER" && \
    usermod -aG sudo "$USER" && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers



# ========================================
FROM system AS python

# Install Python
RUN apt-get update --quiet && \
    apt-get install --quiet --assume-yes --no-install-recommends \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv \
    && \
    apt-get autoremove --quiet --assume-yes && \
    apt-get clean --quiet && \
    rm /var/lib/apt/lists/* --recursive --force

# Install Python version explicitly
ARG VERSION_PYTHON_CONTAINER=3.12
RUN add-apt-repository --yes ppa:deadsnakes/ppa
RUN apt-get update --quiet && \
    apt-get install --quiet --assume-yes --no-install-recommends \
    "python$VERSION_PYTHON_CONTAINER" \
    "python$VERSION_PYTHON_CONTAINER-venv" \
    && \
    apt-get autoremove --quiet --assume-yes && \
    apt-get clean --quiet && \
    rm /var/lib/apt/lists/* --recursive --force

# Configure Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::setuptools.command.develop,ignore:::setuptools.dist,ignore:::pkg_resources \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Setup venv
ARG VERSION_PIP_CONTAINER=25.2 \
    VERSION_SETUPTOOLS_CONTAINER=80.9.0 \
    VERSION_WHEEL_CONTAINER=0.45.1
RUN eval "python$VERSION_PYTHON_CONTAINER" -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/venv/lib/python$VERSION_PYTHON_CONTAINER/site-packages:$PYTHONPATH"
RUN pip install --no-cache-dir \
    "pip==$VERSION_PIP_CONTAINER" \
    "setuptools==$VERSION_SETUPTOOLS_CONTAINER" \
    "wheel==$VERSION_WHEEL_CONTAINER"



# ========================================
FROM python AS libs

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
    rm -rf /tmp/*



# ========================================
FROM libs AS pytorch-project

# Add code
ARG USER=user \
    UID=1001
COPY . "/home/$USER/repos/pytorch-project-template"
RUN chown --recursive "$USER:$USER" "/home/$USER"

# Build
WORKDIR "/home/$USER/repos/pytorch-project-template"
RUN pip install --editable .

# Setup run
USER "$USER"
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash"]
